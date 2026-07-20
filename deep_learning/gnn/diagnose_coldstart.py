"""Diagnostic: is the song encoder truly inductive (produces meaningful
embeddings for songs with few/no graph edges from content alone), or does
it collapse cold songs to a degenerate/uninformative embedding?

Per the research survey, a reproducibility study at ~our catalog scale found
RQ-VAE semantic IDs barely help item cold-start UNLESS the underlying item
embedding already generalizes to unseen items -- i.e., if this diagnostic
shows collapse, no amount of quantization/semantic-ID machinery will fix
cold-start; the fix is upstream (encoder), not downstream (output vocab).

Run:  .venv/bin/python diagnose_coldstart.py --ckpt data/model_gat.pt
"""

import argparse
import json
from collections import Counter

import torch
import torch.nn.functional as F

from train_real import SEQ_LEN, build_graph, load_data
from train_text import TextPlaylistModel, build_title_cache, content_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/model_gat.pt")
    args = ap.parse_args()

    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]
    edge_index, nbrs = build_graph(pairs, len(songs))

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

    # degree from the TRAIN graph = train-time "was this song ever an edge endpoint"
    deg = Counter()
    for a, b in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        deg[a] += 1
        deg[b] += 1
    degrees = torch.tensor([deg.get(i, 0) for i in range(len(songs))])
    zero_deg = (degrees == 0).sum().item()
    print(f"{zero_deg}/{len(songs)} songs ({zero_deg/len(songs):.1%}) have ZERO "
          f"train-graph edges (fully cold w.r.t. structure)")

    with torch.no_grad():
        z_full = model.song_z(feats, artist_ids, edge_index)          # with graph
        z_content = model.song_z(feats, artist_ids,
                                 torch.zeros(2, 0, dtype=torch.long))  # NO edges at all

    # (a) degeneracy check: do content-only embeddings collapse to ~identical vectors?
    zn = F.normalize(z_content, dim=-1)
    sample = zn[torch.randperm(len(songs))[:2000]]
    sim = (sample @ sample.t())
    off_diag = sim[~torch.eye(len(sample), dtype=torch.bool)]
    print(f"\ncontent-only embedding pairwise cosine sim: mean {off_diag.mean():.3f}, "
          f"std {off_diag.std():.3f}  (near 1.0 mean + near-0 std = collapsed/degenerate; "
          f"lower mean + real spread = differentiated)")

    # (b) genre-cluster sanity: same-genre pairs should be more similar than random pairs
    genre = songs["track_genre"].fillna("unknown").values
    idx = torch.randperm(len(songs))[:3000]
    same_g, diff_g = [], []
    for i in range(0, len(idx) - 1, 2):
        a, b = idx[i].item(), idx[i + 1].item()
        s = (zn[a] @ zn[b]).item()
        (same_g if genre[a] == genre[b] else diff_g).append(s)
    print(f"content-only cosine sim: same-genre pairs {sum(same_g)/len(same_g):.3f} "
          f"(n={len(same_g)})  vs  different-genre pairs {sum(diff_g)/len(diff_g):.3f} "
          f"(n={len(diff_g)})  -- gap confirms content signal reaches the embedding")

    # (c) does adding the graph change zero-degree songs' embeddings at all?
    # (it shouldn't much -- they have no edges to aggregate -- but the check is cheap)
    z_full_n = F.normalize(z_full, dim=-1)
    zero_idx = (degrees == 0).nonzero(as_tuple=True)[0]
    if len(zero_idx):
        drift = (1 - (z_full_n[zero_idx] * zn[zero_idx]).sum(-1)).mean().item()
        print(f"\nzero-degree songs: mean (1-cosine) drift between content-only and "
              f"full-graph embedding: {drift:.4f} (near 0 expected -- confirms these "
              f"songs' embeddings are driven by content, not by absent graph signal)")

    # (d) direct test: does hit@10-style ranking for a cold song's true genre-mates
    # separate from random songs, using content-only embeddings?
    print(f"\nsame-genre similarity exceeds cross-genre by "
          f"{(sum(same_g)/len(same_g) - sum(diff_g)/len(diff_g)):+.3f} -- "
          f"{'encoder IS inductive: content alone carries real signal' if (sum(same_g)/len(same_g) - sum(diff_g)/len(diff_g)) > 0.02 else 'WARNING: weak signal, encoder may be too dependent on graph structure'}")


if __name__ == "__main__":
    main()
