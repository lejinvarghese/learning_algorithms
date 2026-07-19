"""Exp 4: do intent-dependent sequential arcs generalize to unseen intents?

Synthetic testbed (real playlists are alphabetized, so arcs must be studied
where true order exists). Arc-bearing intent tokens shape the ground-truth
ENERGY trajectory of a session:

    workout -> ramp up      night -> wind down      dance -> arch (mid peak)
    chill   -> flat low     focus -> flat mid

Queries are (genre, arc-token[, valence mood]); 15% of distinct combos are
held out entirely, so test measures arc generalization to unseen intents.
Compares the plain GRU decoder vs. a phase-conditioned decoder (learned
position embedding added to each step input).

Metrics: teacher-forced hit@10; Pearson corr between generated and target
energy trajectories (dynamic arcs); MAE to target energy (all arcs).

Run:  .venv/bin/python exp_arcs.py
"""

import argparse
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from playlist_gnn import (GENRES, PlaylistDecoder, PlaylistModel, TOK2ID,
                          VOCAB, INTENTS, make_songs, pad_tokens)

L = 10
ARCS = {
    "workout": lambda u: 0.30 + 0.60 * u,
    "night":   lambda u: 0.85 - 0.60 * u,
    "dance":   lambda u: 0.45 + 0.45 * math.sin(math.pi * u),
    "chill":   lambda u: 0.25,
    "focus":   lambda u: 0.45,
}
MOODS = ["sunny", "moody", "happy", "sad"]        # valence-only tokens


def sample_combos(rng):
    combos = [(g, a, m) for g in GENRES for a in ARCS for m in MOODS + [None]]
    rng.shuffle(combos)
    n_test = int(0.15 * len(combos))
    return combos[n_test:], combos[:n_test]


def gen_arc_playlist(feats, meta, combo, rng, temp=0.25):
    genre, arc, mood = combo
    n = feats.size(0)
    gbonus = torch.tensor([2.0 if GENRES[m["genre"]] == genre else 0.0 for m in meta])
    tgt_val = INTENTS[mood]["valence"] if mood else None
    seq = []
    for t in range(L):
        e_t = ARCS[arc](t / (L - 1))
        aff = gbonus - 4.0 * (feats[:, 0] - e_t).abs()
        if tgt_val is not None:
            aff = aff - 2.0 * (feats[:, 1] - tgt_val).abs()
        if seq:
            aff[torch.tensor(seq)] = -1e9
        seq.append(torch.multinomial(F.softmax(aff / temp, 0), 1).item())
    return seq


def make_pairs(feats, meta, combos, n_pairs, rng):
    pairs = []
    for _ in range(n_pairs):
        combo = combos[rng.randrange(len(combos))]
        toks = [w for w in (combo[0], combo[1], combo[2]) if w]
        rng.shuffle(toks)
        pairs.append((toks, gen_arc_playlist(feats, meta, combo, rng), combo))
    return pairs


def build_graph(train_pairs, n):
    counts = Counter()
    for _, seq, _ in train_pairs:
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] += 1
    edges = [e for e, c in counts.items() if c >= 2]
    src, dst = zip(*edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    adj = torch.zeros(n, n)
    adj[edge_index[0], edge_index[1]] = 1.0
    return edge_index, adj


class PhaseDecoder(PlaylistDecoder):
    """Adds a learned per-step phase embedding to the decoder input."""

    def __init__(self, hid):
        super().__init__(hid)
        self.pos = nn.Embedding(L, hid)

    def forward(self, z, tok_emb, tok_mask, pooled, adj, seqs):
        B, T = seqs.shape
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(B, -1)
        last, outs = None, []
        for t in range(T):
            h, logits = self.step(h, inp + self.pos.weight[t], tok_emb, tok_mask,
                                  z, adj, last)
            outs.append(logits)
            last = seqs[:, t]
            inp = z[last]
        return torch.stack(outs, 1)

    @torch.no_grad()
    def generate(self, z, tok_emb, tok_mask, pooled, adj, length=L,
                 temp=0.5, topk=20):
        h = torch.tanh(self.init(pooled))
        inp = self.start.expand(1, -1)
        seq, last = [], None
        for t in range(length):
            h, logits = self.step(h, inp + self.pos.weight[t], tok_emb, tok_mask,
                                  z, adj, last)
            logits = logits.squeeze(0)
            if seq:
                logits[torch.tensor(seq)] = -1e9
            kth = logits.topk(topk).values[-1]
            logits[logits < kth] = -1e9
            nxt = torch.multinomial(F.softmax(logits / temp, 0), 1).item()
            seq.append(nxt)
            last = torch.tensor([nxt])
            inp = z[last]
        return seq


def train_model(model, pairs, feats, edge_index, adj, epochs=8, batch=64):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        random.shuffle(pairs)
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            tokens = pad_tokens([p[0] for p in chunk])
            seqs = torch.tensor([p[1] for p in chunk])
            z = model.songs(feats, edge_index)
            tok_emb, tok_mask, pooled = model.query(tokens)
            logits = model.dec(z, tok_emb, tok_mask, pooled, adj, seqs)
            loss = F.cross_entropy(logits.flatten(0, 1), seqs.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, pairs, feats, edge_index, adj, n_gen=150):
    model.eval()
    z = model.songs(feats, edge_index)
    # teacher-forced hit@10
    tokens = pad_tokens([p[0] for p in pairs])
    seqs = torch.tensor([p[1] for p in pairs])
    tok_emb, tok_mask, pooled = model.query(tokens)
    logits = model.dec(z, tok_emb, tok_mask, pooled, adj, seqs)
    hit = (logits.topk(10, -1).indices == seqs.unsqueeze(-1)).any(-1).float().mean().item()
    # arc fidelity on generated playlists
    corrs, maes = [], []
    for toks, _, combo in pairs[:n_gen]:
        tk = pad_tokens([toks])
        te, tm, pl = model.query(tk)
        seq = model.dec.generate(z, te, tm, pl, adj)
        e = feats[torch.tensor(seq), 0]
        tgt = torch.tensor([ARCS[combo[1]](t / (L - 1)) for t in range(L)])
        maes.append((e - tgt).abs().mean().item())
        if tgt.std() > 0.05:                       # corr undefined for flat arcs
            corrs.append(torch.corrcoef(torch.stack([e, tgt]))[0, 1].item())
    return hit, sum(corrs) / len(corrs), sum(maes) / len(maes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    feats, meta = make_songs(500, rng)
    train_combos, test_combos = sample_combos(rng)
    train_pairs = make_pairs(feats, meta, train_combos, 3000, rng)
    test_pairs = make_pairs(feats, meta, test_combos, 400, rng)
    edge_index, adj = build_graph(train_pairs, 500)
    print(f"{edge_index.size(1)} edges, {len(train_combos)} train / "
          f"{len(test_combos)} held-out intent combos")

    for name, dec in [("baseline GRU", None), ("phase-conditioned", PhaseDecoder(64))]:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        model = PlaylistModel(feats.size(1), 64, len(VOCAB))
        if dec is not None:
            model.dec = dec
        train_model(model, list(train_pairs), feats, edge_index, adj)
        hit, corr, mae = evaluate(model, test_pairs, feats, edge_index, adj)
        print(f"{name:<18} unseen-combo hit@10 {hit:.1%}  "
              f"arc corr {corr:+.3f}  energy MAE {mae:.3f}")

    # show one unseen-combo trajectory
    model.eval()
    z = model.songs(feats, edge_index)
    combo = next(c for c in test_combos if c[1] in ("workout", "night", "dance"))
    toks = [w for w in combo if w]
    tk = pad_tokens([toks])
    te, tm, pl = model.query(tk)
    seq = model.dec.generate(z, te, tm, pl, adj)
    e = [round(feats[s, 0].item(), 2) for s in seq]
    tgt = [round(ARCS[combo[1]](t / (L - 1)), 2) for t in range(L)]
    print(f"\nunseen combo {toks} (phase model)\n  target energy: {tgt}\n  generated:     {e}")


if __name__ == "__main__":
    main()
