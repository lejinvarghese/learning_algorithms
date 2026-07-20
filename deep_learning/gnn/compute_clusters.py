"""Precompute 2-level residual k-means cluster IDs from a frozen checkpoint's
song embeddings — a cheap "RQ-KMeans" proxy (per the research survey) used
as an auxiliary training target, NOT as the decoding vocabulary. Diagnostic
(diagnose_coldstart.py) confirmed our encoder is genuinely inductive (cold
songs get real content-driven embeddings, not degenerate ones), so the
question this tests is narrower than full RQ-VAE/TIGER: does giving the
decoder an extra "which neighborhood is the answer in" signal help ranking
calibration on rare songs, at near-zero implementation cost.

Run:  .venv/bin/python compute_clusters.py --ckpt data/model_gat.pt
"""

import argparse
import json

import numpy as np
import torch
from sklearn.cluster import KMeans

from train_real import SEQ_LEN, build_graph, load_data
from train_text import TextPlaylistModel, build_title_cache, content_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/model_gat.pt")
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=32)
    ap.add_argument("--out", default="data/clusters.pt")
    args = ap.parse_args()

    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]
    edge_index, _ = build_graph(pairs, len(songs))

    ckpt = torch.load(args.ckpt)
    lex_vocab = ckpt.get("lex_vocab", [])
    artist_ids = None
    if ckpt.get("content"):
        fa = songs["artists"].fillna("").str.split(";").str[0].str.lower()
        a2i = {a: i for i, a in enumerate(sorted(set(fa)))}
        artist_ids = torch.tensor([a2i[a] for a in fa])
        feats = torch.cat([feats, content_features(songs), build_title_cache(songs)], 1)
    lex_init = (torch.zeros(len(lex_vocab) + 1, ckpt["emb_dim"])
                if ckpt.get("lexv2") else None)
    model = TextPlaylistModel(feats.size(1), ckpt["emb_dim"], ckpt["hidden"],
                              len(lex_vocab), ckpt.get("n_artists", 0),
                              ckpt.get("layers", 0), lex_init,
                              ckpt.get("steer", False), ckpt.get("gat", False))
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    with torch.no_grad():
        z = model.song_z(feats, artist_ids, edge_index).numpy()

    print(f"level 1: k-means K={args.k1} on {z.shape[0]} songs, dim {z.shape[1]}")
    km1 = KMeans(n_clusters=args.k1, n_init=10, random_state=7).fit(z)
    c1 = km1.labels_
    print(f"  usage: min {np.bincount(c1).min()}, max {np.bincount(c1).max()}, "
          f"mean {np.bincount(c1).mean():.1f} (uniform would be {len(c1)/args.k1:.1f})")

    residual = z - km1.cluster_centers_[c1]
    print(f"level 2 (residual): k-means K={args.k2}")
    km2 = KMeans(n_clusters=args.k2, n_init=10, random_state=7).fit(residual)
    c2 = km2.labels_
    print(f"  usage: min {np.bincount(c2).min()}, max {np.bincount(c2).max()}, "
          f"mean {np.bincount(c2).mean():.1f} (uniform would be {len(c2)/args.k2:.1f})")

    combo = list(zip(c1.tolist(), c2.tolist()))
    print(f"combined (c1,c2) pairs: {len(set(combo))} unique / {len(combo)} songs "
          f"({len(set(combo))/len(combo):.1%} unique)")

    torch.save({"c1": torch.tensor(c1, dtype=torch.long),
                "c2": torch.tensor(c2, dtype=torch.long),
                "k1": args.k1, "k2": args.k2, "source_ckpt": args.ckpt}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
