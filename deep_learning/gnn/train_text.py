"""Train the playlist model with frozen ModernBERT text embeddings as the
query encoder (Alibaba-NLP/gte-modernbert-base via sentence-transformers).

Replaces the closed 195-token vocab: every playlist name becomes a usable
query (playlists_all.jsonl, no vocab filter), and arbitrary inference-time
words land in the same embedding space. Query tokens = [whole-name sentence
embedding] + per-word embeddings, feeding the same cross-attention decoder.

Run:  .venv/bin/python train_text.py
"""

import argparse
import json
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from playlist_gnn import SongEncoder
from train_real import SEQ_LEN, SparseNeighborDecoder, build_graph, load_data

EMB_MODEL = "Alibaba-NLP/gte-modernbert-base"
MAX_WORDS = 7            # query tokens = 1 sentence emb + up to 7 word embs
CACHE = "data/text_cache.pt"


def words_of(name):
    seen, out = set(), []
    for w in re.findall(r"[a-z0-9]+", name.lower()):
        if len(w) >= 2 and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)


def build_text_cache(names):
    if os.path.exists(CACHE):
        return torch.load(CACHE)
    enc = get_encoder()
    words = sorted({w for n in names for w in words_of(n)})
    print(f"encoding {len(names)} names + {len(words)} words with {EMB_MODEL} ...")
    name_emb = torch.tensor(enc.encode(names, batch_size=128, normalize_embeddings=True))
    word_emb = torch.tensor(enc.encode(words, batch_size=256, normalize_embeddings=True))
    cache = {"names": names, "name_emb": name_emb, "words": words, "word_emb": word_emb}
    torch.save(cache, CACHE)
    return cache


class TextQueryEncoder(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)

    def forward(self, emb, mask):
        tok = self.proj(emb)
        pooled = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return tok, mask, pooled


class TextPlaylistModel(nn.Module):
    def __init__(self, feat_dim, emb_dim, hid):
        super().__init__()
        self.songs = SongEncoder(feat_dim, hid)
        self.query = TextQueryEncoder(emb_dim, hid)
        self.dec = SparseNeighborDecoder(hid)


def query_embs_for(name, cache, n2i, w2i):
    """[T, D] frozen embeddings + mask for one query string."""
    D = cache["name_emb"].size(1)
    emb = torch.zeros(1 + MAX_WORDS, D)
    mask = torch.zeros(1 + MAX_WORDS, dtype=torch.bool)
    emb[0] = cache["name_emb"][n2i[name]]
    mask[0] = True
    for j, w in enumerate(words_of(name)[:MAX_WORDS]):
        emb[1 + j] = cache["word_emb"][w2i[w]]
        mask[1 + j] = True
    return emb, mask


def encode_batch(chunk, cache, n2i, w2i):
    embs, masks = zip(*(query_embs_for(p["name"], cache, n2i, w2i) for p in chunk))
    seqs = torch.tensor([p["track_ids"][:SEQ_LEN] for p in chunk])
    return torch.stack(embs), torch.stack(masks), seqs


def run_epoch(model, opt, pairs, feats, edge_index, nbrs, cache, n2i, w2i, batch=64):
    model.train()
    random.shuffle(pairs)
    total, nb = 0.0, 0
    for i in range(0, len(pairs), batch):
        emb, mask, seqs = encode_batch(pairs[i:i + batch], cache, n2i, w2i)
        z = model.songs(feats, edge_index)
        tok, tmask, pooled = model.query(emb, mask)
        logits = model.dec(z, tok, tmask, pooled, nbrs, seqs)
        loss = F.cross_entropy(logits.flatten(0, 1), seqs.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, nb = total + loss.item(), nb + 1
    return total / nb


@torch.no_grad()
def hit_at_k(model, pairs, feats, edge_index, nbrs, cache, n2i, w2i, k=10, batch=256):
    model.eval()
    z = model.songs(feats, edge_index)
    hits, n = 0, 0
    for i in range(0, len(pairs), batch):
        emb, mask, seqs = encode_batch(pairs[i:i + batch], cache, n2i, w2i)
        tok, tmask, pooled = model.query(emb, mask)
        logits = model.dec(z, tok, tmask, pooled, nbrs, seqs)
        topk = logits.topk(k, dim=-1).indices
        hits += (topk == seqs.unsqueeze(-1)).any(-1).sum().item()
        n += seqs.numel()
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]

    names = sorted({p["name"] for p in pairs})
    cache = build_text_cache(names)
    n2i = {n: i for i, n in enumerate(cache["names"])}
    w2i = {w: i for i, w in enumerate(cache["words"])}
    emb_dim = cache["name_emb"].size(1)

    random.shuffle(pairs)
    n_test = len(pairs) // 10
    test, train = pairs[:n_test], pairs[n_test:]
    # graph from training playlists only — no test co-listen edges leak in
    edge_index, nbrs = build_graph(train, len(songs))
    print(f"{len(songs)} songs, {edge_index.size(1)} edges (train-only), "
          f"{len(train)} train / {len(test)} test pairs, emb dim {emb_dim}")

    model = TextPlaylistModel(feats.size(1), emb_dim, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        loss = run_epoch(model, opt, train, feats, edge_index, nbrs, cache, n2i, w2i)
        hit = hit_at_k(model, test, feats, edge_index, nbrs, cache, n2i, w2i)
        print(f"epoch {ep+1}: loss {loss:.3f}  test hit@10 {hit:.2%} "
              f"(random {10/len(songs):.2%})")

    torch.save({"model": model.state_dict(), "emb_dim": emb_dim,
                "hidden": args.hidden}, "data/model_text.pt")
    print("saved data/model_text.pt")


if __name__ == "__main__":
    main()
