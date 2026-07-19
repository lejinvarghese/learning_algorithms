"""Train the query-conditioned playlist model on real data:
Zenodo #nowplaying Spotify playlists x HF 114k-track audio features.

Queries = tokens mined from real playlist names. Graph = within-playlist
co-listening edges (this dump alphabetizes playlists, so edges are
co-membership, not true play transitions).

Run:  .venv/bin/python train_real.py
"""

import argparse
import json
import random
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F

from playlist_gnn import PlaylistDecoder, PlaylistModel

SCALARS = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
           "speechiness", "liveness"]
SEQ_LEN = 10
N_GENRES = 30


class SparseNeighborDecoder(PlaylistDecoder):
    """Same decoder, but the graph bias uses per-node neighbor lists instead of
    a dense NxN adjacency (11k+ songs)."""

    def step(self, h, inp, tok_emb, tok_mask, z, nbrs, last):
        h = self.gru(inp, h)
        c, _ = self.attn(h.unsqueeze(1), tok_emb, tok_emb, key_padding_mask=~tok_mask)
        p = self.out(torch.cat([h, c.squeeze(1)], -1))
        logits = p @ z.t()
        if last is not None:
            bonus = torch.zeros_like(logits)
            for b, l in enumerate(last.tolist()):
                if nbrs[l].numel():
                    bonus[b, nbrs[l]] = 1.0
            logits = logits + self.edge_bias * bonus
        return h, logits


def load_data():
    songs = pd.read_parquet("data/songs_real.parquet")
    pairs = [json.loads(l) for l in open("data/playlists_real.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]

    top_genres = songs["track_genre"].value_counts().head(N_GENRES).index.tolist()
    g2i = {g: i for i, g in enumerate(top_genres)}
    feats = torch.zeros(len(songs), len(SCALARS) + 3 + N_GENRES)
    for j, c in enumerate(SCALARS):
        feats[:, j] = torch.tensor(songs[c].values, dtype=torch.float)
    feats[:, len(SCALARS)] = torch.tensor((songs["tempo"].values / 250).clip(0, 1), dtype=torch.float)
    feats[:, len(SCALARS) + 1] = torch.tensor(((songs["loudness"].values + 60) / 60).clip(0, 1), dtype=torch.float)
    feats[:, len(SCALARS) + 2] = torch.tensor(songs["popularity"].values / 100, dtype=torch.float)
    for i, g in enumerate(songs["track_genre"]):
        if g in g2i:
            feats[i, len(SCALARS) + 3 + g2i[g]] = 1.0

    tok_freq = Counter(t for p in pairs for t in p["tokens"])
    vocab = ["<pad>"] + [t for t, _ in tok_freq.most_common()]
    tok2id = {t: i for i, t in enumerate(vocab)}
    return songs, pairs, feats, vocab, tok2id


def build_graph(pairs, n_songs, min_count=2):
    counts = Counter()
    for p in pairs:
        ids = p["track_ids"]
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
    edges = [e for e, c in counts.items() if c >= min_count]
    src, dst = zip(*edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    nbrs = [[] for _ in range(n_songs)]
    for a, b in edges:
        nbrs[a].append(b)
    nbrs = [torch.tensor(x, dtype=torch.long) for x in nbrs]
    return edge_index, nbrs


def encode_batch(chunk, tok2id, max_tok=6):
    tokens = torch.zeros(len(chunk), max_tok, dtype=torch.long)
    for i, p in enumerate(chunk):
        ids = [tok2id[t] for t in p["tokens"]][:max_tok]
        tokens[i, :len(ids)] = torch.tensor(ids)
    seqs = torch.tensor([p["track_ids"][:SEQ_LEN] for p in chunk])
    return tokens, seqs


def run_epoch(model, opt, pairs, feats, edge_index, nbrs, tok2id, batch=64):
    model.train()
    random.shuffle(pairs)
    total, nb = 0.0, 0
    for i in range(0, len(pairs), batch):
        tokens, seqs = encode_batch(pairs[i:i + batch], tok2id)
        z = model.songs(feats, edge_index)
        tok_emb, tok_mask, pooled = model.query(tokens)
        logits = model.dec(z, tok_emb, tok_mask, pooled, nbrs, seqs)
        loss = F.cross_entropy(logits.flatten(0, 1), seqs.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total, nb = total + loss.item(), nb + 1
    return total / nb


@torch.no_grad()
def hit_at_k(model, pairs, feats, edge_index, nbrs, tok2id, k=10, batch=256):
    model.eval()
    z = model.songs(feats, edge_index)
    hits, n = 0, 0
    for i in range(0, len(pairs), batch):
        tokens, seqs = encode_batch(pairs[i:i + batch], tok2id)
        tok_emb, tok_mask, pooled = model.query(tokens)
        logits = model.dec(z, tok_emb, tok_mask, pooled, nbrs, seqs)
        topk = logits.topk(k, dim=-1).indices
        hits += (topk == seqs.unsqueeze(-1)).any(-1).sum().item()
        n += seqs.numel()
    return hits / n


@torch.no_grad()
def demo(model, query_tokens, songs, feats, edge_index, nbrs, tok2id):
    model.eval()
    toks = [t for t in query_tokens if t in tok2id]
    if not toks:
        print(f"  (no known tokens in {query_tokens})")
        return
    z = model.songs(feats, edge_index)
    tokens = torch.zeros(1, 6, dtype=torch.long)
    ids = [tok2id[t] for t in toks][:6]
    tokens[0, :len(ids)] = torch.tensor(ids)
    tok_emb, tok_mask, pooled = model.query(tokens)
    seq = model.dec.generate(z, tok_emb, tok_mask, pooled, nbrs, length=SEQ_LEN)
    print(f"\nquery tokens: {toks}")
    for i, s in enumerate(seq):
        r = songs.iloc[s]
        print(f"  {i+1:>2}  {r['artists'].split(';')[0][:24]:<24} - {r['track_name'][:34]:<34} "
              f"[{r['track_genre']}] en={r['energy']:.2f} va={r['valence']:.2f} da={r['danceability']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    songs, pairs, feats, vocab, tok2id = load_data()
    edge_index, nbrs = build_graph(pairs, len(songs))
    print(f"{len(songs)} songs, {edge_index.size(1)} co-listen edges "
          f"(density {edge_index.size(1)/len(songs)**2:.4%}), "
          f"{len(pairs)} pairs, vocab {len(vocab)}")

    random.shuffle(pairs)
    n_test = len(pairs) // 10
    test, train = pairs[:n_test], pairs[n_test:]

    model = PlaylistModel(feats.size(1), args.hidden, len(vocab))
    model.dec = SparseNeighborDecoder(args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rand_base = 10 / len(songs)
    for ep in range(args.epochs):
        loss = run_epoch(model, opt, train, feats, edge_index, nbrs, tok2id)
        hit = hit_at_k(model, test, feats, edge_index, nbrs, tok2id)
        print(f"epoch {ep+1}: loss {loss:.3f}  test hit@10 {hit:.2%} (random {rand_base:.2%})")

    combos = {frozenset(p["tokens"]) for p in train}
    seen = [p for p in test if frozenset(p["tokens"]) in combos]
    unseen = [p for p in test if frozenset(p["tokens"]) not in combos]
    if seen and unseen:
        print(f"hit@10 seen combos ({len(seen)}): "
              f"{hit_at_k(model, seen, feats, edge_index, nbrs, tok2id):.2%}   "
              f"unseen combos ({len(unseen)}): "
              f"{hit_at_k(model, unseen, feats, edge_index, nbrs, tok2id):.2%}")

    torch.save({"model": model.state_dict(), "vocab": vocab}, "data/model_real.pt")

    for q in [["chill", "acoustic"], ["christmas"], ["summer", "party"],
              ["rock", "metal"], ["hip", "hop", "party"], ["punk", "rock", "chill"]]:
        demo(model, q, songs, feats, edge_index, nbrs, tok2id)


if __name__ == "__main__":
    main()
