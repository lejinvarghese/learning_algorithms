"""Train the playlist model with frozen ModernBERT text embeddings as the
query encoder (Alibaba-NLP/gte-modernbert-base via sentence-transformers).

Query tokens = [whole-name sentence embedding] + per-word dense embeddings,
optionally + learned exact-match lexical token embeddings (--lexical), all
feeding the same cross-attention decoder.

--coverage W adds an attention-coverage loss: every real query token must
accumulate attention mass across the decode steps, penalizing "unmet
intents" (dominant-facet capture).

Run:  .venv/bin/python train_text.py --lexical [--coverage 1.0]
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from playlist_gnn import SongEncoder
from train_real import SEQ_LEN, SparseNeighborDecoder, build_graph, load_data

EMB_MODEL = "Alibaba-NLP/gte-modernbert-base"
MAX_WORDS = 7            # dense tokens = 1 sentence emb + up to 7 word embs
MAX_LEX = 5              # exact-match lexical tokens per query
COV_TAU = 0.4            # min attention mass a real token should accumulate
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


def build_lex_vocab(train_pairs, min_freq=40, cap=300):
    cnt = Counter(w for p in train_pairs for w in set(words_of(p["name"])))
    return [w for w, c in cnt.most_common(cap) if c >= min_freq]


class TextQueryEncoder(nn.Module):
    """Dense (frozen, projected) tokens + optional learned lexical tokens."""

    def __init__(self, in_dim, hid, lex_vocab=0):
        super().__init__()
        self.proj = nn.Linear(in_dim, hid)
        self.lex = nn.Embedding(lex_vocab + 1, hid, padding_idx=0) if lex_vocab else None

    def forward(self, emb, mask, lex_ids=None):
        tok = self.proj(emb)
        if self.lex is not None and lex_ids is not None:
            tok = torch.cat([tok, self.lex(lex_ids)], 1)
            mask = torch.cat([mask, lex_ids != 0], 1)
        pooled = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return tok, mask, pooled


class HybridDecoder(SparseNeighborDecoder):
    """Adds attention-coverage tracking and optional MMR-diverse generation."""

    def step(self, h, inp, tok_emb, tok_mask, z, nbrs, last):
        h = self.gru(inp, h)
        c, attn = self.attn(h.unsqueeze(1), tok_emb, tok_emb,
                            key_padding_mask=~tok_mask, need_weights=True)
        p = self.out(torch.cat([h, c.squeeze(1)], -1))
        logits = p @ z.t()
        if last is not None:
            bonus = torch.zeros_like(logits)
            for b, l in enumerate(last.tolist()):
                if nbrs[l].numel():
                    bonus[b, nbrs[l]] = 1.0
            logits = logits + self.edge_bias * bonus
        return h, logits, attn.squeeze(1)          # attn: [B, T]

    def forward(self, z, tok_emb, tok_mask, pooled, nbrs, seqs):
        B, L = seqs.shape
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(B, -1)
        last, all_logits, cov = None, [], torch.zeros(B, tok_emb.size(1))
        for t in range(L):
            h, logits, attn = self.step(h, inp, tok_emb, tok_mask, z, nbrs, last)
            all_logits.append(logits)
            cov = cov + attn
            last = seqs[:, t]
            inp = z[last]
        return torch.stack(all_logits, 1), cov

    @torch.no_grad()
    def generate(self, z, tok_emb, tok_mask, pooled, nbrs, length=SEQ_LEN,
                 temp=0.7, topk=20, mmr=0.0):
        zn = F.normalize(z, dim=-1)
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(1, -1)
        seq, last = [], None
        for _ in range(length):
            h, logits, _ = self.step(h, inp, tok_emb, tok_mask, z, nbrs, last)
            logits = logits.squeeze(0)
            if seq:
                logits[torch.tensor(seq)] = -1e9
                if mmr > 0:                        # penalize similarity to picks
                    sim = (zn @ zn[torch.tensor(seq)].t()).max(dim=1).values
                    logits = logits - mmr * logits.abs().mean() * sim
            kth = logits.topk(topk).values[-1]
            logits[logits < kth] = -1e9
            nxt = torch.multinomial(F.softmax(logits / temp, 0), 1).item()
            seq.append(nxt)
            last = torch.tensor([nxt])
            inp = z[last]
        return seq


class TextPlaylistModel(nn.Module):
    def __init__(self, feat_dim, emb_dim, hid, lex_vocab=0):
        super().__init__()
        self.songs = SongEncoder(feat_dim, hid)
        self.query = TextQueryEncoder(emb_dim, hid, lex_vocab)
        self.dec = HybridDecoder(hid)


def query_embs_for(name, cache, n2i, w2i, lex2id):
    D = cache["name_emb"].size(1)
    emb = torch.zeros(1 + MAX_WORDS, D)
    mask = torch.zeros(1 + MAX_WORDS, dtype=torch.bool)
    emb[0] = cache["name_emb"][n2i[name]]
    mask[0] = True
    ws = words_of(name)
    for j, w in enumerate(ws[:MAX_WORDS]):
        emb[1 + j] = cache["word_emb"][w2i[w]]
        mask[1 + j] = True
    lex = torch.zeros(MAX_LEX, dtype=torch.long)
    if lex2id:
        hits = [lex2id[w] for w in ws if w in lex2id][:MAX_LEX]
        lex[:len(hits)] = torch.tensor(hits)
    return emb, mask, lex


def encode_batch(chunk, cache, n2i, w2i, lex2id):
    embs, masks, lexs = zip(*(query_embs_for(p["name"], cache, n2i, w2i, lex2id)
                              for p in chunk))
    seqs = torch.tensor([p["track_ids"][:SEQ_LEN] for p in chunk])
    return torch.stack(embs), torch.stack(masks), torch.stack(lexs), seqs


def coverage_loss(cov, mask):
    """Penalty for real query tokens left under-attended after all steps."""
    unmet = F.relu(COV_TAU - cov) * mask
    return unmet.sum() / mask.sum().clamp(min=1)


def run_epoch(model, opt, pairs, feats, edge_index, nbrs, cache, n2i, w2i,
              lex2id, cov_w, batch=64):
    model.train()
    random.shuffle(pairs)
    total, nb = 0.0, 0
    for i in range(0, len(pairs), batch):
        emb, mask, lex, seqs = encode_batch(pairs[i:i + batch], cache, n2i, w2i, lex2id)
        z = model.songs(feats, edge_index)
        tok, tmask, pooled = model.query(emb, mask, lex)
        logits, cov = model.dec(z, tok, tmask, pooled, nbrs, seqs)
        loss = F.cross_entropy(logits.flatten(0, 1), seqs.flatten())
        if cov_w > 0:
            loss = loss + cov_w * coverage_loss(cov, tmask.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, nb = total + loss.item(), nb + 1
    return total / nb


@torch.no_grad()
def evaluate(model, pairs, feats, edge_index, nbrs, cache, n2i, w2i, lex2id,
             k=10, batch=256):
    """Returns (hit@k, unmet-intent rate: frac of real tokens with cov<0.2)."""
    model.eval()
    z = model.songs(feats, edge_index)
    hits = n = unmet = ntok = 0
    for i in range(0, len(pairs), batch):
        emb, mask, lex, seqs = encode_batch(pairs[i:i + batch], cache, n2i, w2i, lex2id)
        tok, tmask, pooled = model.query(emb, mask, lex)
        logits, cov = model.dec(z, tok, tmask, pooled, nbrs, seqs)
        topk = logits.topk(k, dim=-1).indices
        hits += (topk == seqs.unsqueeze(-1)).any(-1).sum().item()
        n += seqs.numel()
        unmet += ((cov < 0.2) & tmask).sum().item()
        ntok += tmask.sum().item()
    return hits / n, unmet / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lexical", action="store_true")
    ap.add_argument("--coverage", type=float, default=0.0)
    ap.add_argument("--out", default="data/model_text.pt")
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
    edge_index, nbrs = build_graph(train, len(songs))

    lex_vocab = build_lex_vocab(train) if args.lexical else []
    lex2id = {w: i + 1 for i, w in enumerate(lex_vocab)}
    print(f"{len(songs)} songs, {edge_index.size(1)} edges (train-only), "
          f"{len(train)} train / {len(test)} test, emb dim {emb_dim}, "
          f"lexical vocab {len(lex_vocab)}, coverage weight {args.coverage}")

    model = TextPlaylistModel(feats.size(1), emb_dim, args.hidden, len(lex_vocab))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        loss = run_epoch(model, opt, train, feats, edge_index, nbrs, cache,
                         n2i, w2i, lex2id, args.coverage)
        hit, unmet = evaluate(model, test, feats, edge_index, nbrs, cache,
                              n2i, w2i, lex2id)
        print(f"epoch {ep+1}: loss {loss:.3f}  test hit@10 {hit:.2%}  "
              f"unmet-intent rate {unmet:.1%}  (random {10/len(songs):.2%})")

    torch.save({"model": model.state_dict(), "emb_dim": emb_dim,
                "hidden": args.hidden, "lex_vocab": lex_vocab}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
