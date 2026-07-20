"""Systematic failure analysis of the latest model on the exact test split.

Slices teacher-forced hit@10 by: position in playlist, query informativeness,
target-song train popularity, graph-edge presence from previous song, and
artist continuity. Also reports rank stats and concrete failure examples.

Run:  .venv/bin/python analyze_failures.py --ckpt data/model_large.pt
"""

import argparse
import json
import random
from collections import Counter

import torch

from train_real import SEQ_LEN, build_graph, load_data
from train_text import TextPlaylistModel, build_text_cache, encode_batch

JUNK = set("""starred liked radio songs music playlist playlists list lista mix
misc random stuff various artists artist album albums favourite favourites
favorite favorites favoritas favs faves fav spotify shazam soundrop iphone
library play top best new old good great my the and you for jan feb mar apr
may jun jul aug sep oct nov dec january february march april june july august
september october november december monday tuesday wednesday thursday friday
saturday sunday week weekend 2010 2011 2012 2013 2014 2015""".split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/model_large.pt")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # reproduce the exact training split (same seed, same order of RNG calls)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]
    names = sorted({p["name"] for p in pairs})
    cache = build_text_cache(names)
    n2i = {n: i for i, n in enumerate(cache["names"])}
    w2i = {w: i for i, w in enumerate(cache["words"])}
    random.shuffle(pairs)
    n_test = len(pairs) // 10
    test, train = pairs[:n_test], pairs[n_test:]
    edge_index, nbrs = build_graph(train, len(songs))

    ckpt = torch.load(args.ckpt)
    lex_vocab = ckpt.get("lex_vocab", [])
    lex2id = ckpt.get("lexmap") or (
        {"words": {w: i + 1 for i, w in enumerate(lex_vocab)}, "phrases": {}}
        if lex_vocab else {})
    artist_ids = None
    if ckpt.get("content"):
        from train_text import build_title_cache, content_features
        fa = songs["artists"].fillna("").str.split(";").str[0].str.lower()
        a2i = {a: i for i, a in enumerate(sorted(set(fa)))}
        artist_ids = torch.tensor([a2i[a] for a in fa])
        feats = torch.cat([feats, content_features(songs), build_title_cache(songs)], 1)
    lex_init = (torch.zeros(len(lex_vocab) + 1, ckpt["emb_dim"])
                if ckpt.get("lexv2") else None)
    model = TextPlaylistModel(feats.size(1), ckpt["emb_dim"], ckpt["hidden"],
                              len(lex_vocab), ckpt.get("n_artists", 0),
                              ckpt.get("layers", 0), lex_init,
                              ckpt.get("steer", False), ckpt.get("genrebias", False),
                              ckpt.get("gat", False))
    song_genre_ids = None
    if ckpt.get("genrebias"):
        lm = ckpt.get("lexmap", {})
        genres = [v.split("genre:", 1)[1] for v in lex_vocab if v.startswith("genre:")]
        g2idx = {g: i for i, g in enumerate(genres)}
        song_genre_ids = torch.tensor(
            [g2idx.get(g, 0) for g in songs["track_genre"].fillna(genres[0])])
    model.load_state_dict(ckpt["model"])
    model.eval()

    pop = Counter(s for p in train for s in p["track_ids"])
    nbr_sets = [set(t.tolist()) for t in nbrs]
    first_artist = songs["artists"].fillna("").str.split(";").str[0].str.lower()

    def semantic(p):
        ws = [w for w in p["name"].lower().split() if w.isalpha()]
        return any(w not in JUNK and len(w) > 2 for w in ws)

    rows = []          # one dict per (playlist, position)
    examples = []
    with torch.no_grad():
        z = model.song_z(feats, artist_ids, edge_index)
        for i in range(0, len(test), 256):
            chunk = test[i:i + 256]
            emb, mask, lex, seqs, qgenre = encode_batch(chunk, cache, n2i, w2i, lex2id)
            tok, tmask, pooled = model.query(emb, mask, lex)
            logits, _ = model.dec(z, tok, tmask, pooled, nbrs, seqs,
                                  song_genre_ids, qgenre)
            tgt_logit = logits.gather(-1, seqs.unsqueeze(-1))
            rank = (logits > tgt_logit).sum(-1) + 1          # [B, L]
            top10 = logits.topk(10, -1).indices
            for b, p in enumerate(chunk):
                ids = p["track_ids"][:SEQ_LEN]
                for t in range(SEQ_LEN):
                    tgt, prev = ids[t], (ids[t - 1] if t else None)
                    rows.append(dict(
                        t=t, hit=int(rank[b, t] <= 10), rank=rank[b, t].item(),
                        sem=semantic(p), pop=pop.get(tgt, 0),
                        edge=(prev is not None and tgt in nbr_sets[prev]),
                        same_artist=(prev is not None and
                                     first_artist.iloc[tgt] == first_artist.iloc[prev]),
                        remaining_in_top10=len(set(ids[t:]) & set(top10[b, t].tolist())),
                        n_remaining=SEQ_LEN - t))
                if len(examples) < 6 and semantic(p) and rank[b, 5] > 100:
                    examples.append((p, [songs.iloc[s] for s in ids[2:5]],
                                     [songs.iloc[s] for s in top10[b, 5][:3].tolist()],
                                     songs.iloc[ids[5]], rank[b, 5].item()))

    def agg(sel, label):
        sel = list(sel)
        if not sel:
            return
        h = sum(r["hit"] for r in sel) / len(sel)
        med = sorted(r["rank"] for r in sel)[len(sel) // 2]
        print(f"  {label:<38} n={len(sel):<6} hit@10 {h:6.1%}  median rank {med}")

    print(f"\noverall: {len(test)} playlists x {SEQ_LEN} positions")
    agg(rows, "ALL")
    print("\nby position:")
    for t in range(SEQ_LEN):
        agg((r for r in rows if r["t"] == t), f"t={t}" + ("  (seed: query only)" if t == 0 else ""))
    print("\nby query informativeness:")
    agg((r for r in rows if r["sem"]), "semantic name")
    agg((r for r in rows if not r["sem"]), "junk name (Starred/Liked/dates/...)")
    print("\nby target-song popularity in train:")
    for lo, hi in [(0, 5), (5, 20), (20, 100), (100, 10**9)]:
        agg((r for r in rows if lo <= r["pop"] < hi), f"pop {lo}-{hi if hi < 10**9 else 'inf'}")
    print("\nby graph edge from previous song (t>=1):")
    agg((r for r in rows if r["t"] and r["edge"]), "edge exists prev->target")
    agg((r for r in rows if r["t"] and not r["edge"]), "no edge prev->target")
    print("\nby artist continuity (t>=1):")
    agg((r for r in rows if r["t"] and r["same_artist"]), "same artist as previous")
    agg((r for r in rows if r["t"] and not r["same_artist"]), "different artist")
    srec = [r["remaining_in_top10"] / min(10, r["n_remaining"]) for r in rows]
    print(f"\nset recall: top-10 contains {sum(srec)/len(srec):.1%} of the playlist's "
          f"remaining true songs (upper-bounds next-song hit)")

    print("\n--- concrete failures (semantic query, miss at t=5, rank>100) ---")
    for p, prev, pred, tgt, rk in examples[:4]:
        print(f"\nquery: {p['name']!r}")
        print("  context: " + "; ".join(f"{r['artists'].split(';')[0]} - {r['track_name'][:28]}" for r in prev))
        print(f"  TARGET (rank {rk}): {tgt['artists'].split(';')[0]} - {tgt['track_name'][:40]} [{tgt['track_genre']}]")
        print("  predicted: " + "; ".join(f"{r['artists'].split(';')[0]} - {r['track_name'][:28]}" for r in pred))


if __name__ == "__main__":
    main()
