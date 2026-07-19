"""Run free-text queries against the trained real-data model.

Usage:  .venv/bin/python infer_real.py "punk songs for a christmas dinner" ...
With no args, runs a battery of intentionally conflicting queries.
"""

import re
import sys

import torch

from playlist_gnn import PlaylistModel
from train_real import SEQ_LEN, SparseNeighborDecoder, build_graph, load_data

DEFAULT_QUERIES = [
    "punk songs for a christmas dinner",
    "country ballads praising the sensual 70s with anti vietnam war sentiment",
    "sad songs for a summer party",
    "heavy metal love songs",
    "acoustic hip hop chill",
    "christmas metal",
]


def main():
    songs, pairs, feats, _, _ = load_data()
    edge_index, nbrs = build_graph(pairs, len(songs))
    ckpt = torch.load("data/model_real.pt")
    vocab = ckpt["vocab"]
    tok2id = {t: i for i, t in enumerate(vocab)}
    model = PlaylistModel(feats.size(1), 64, len(vocab))
    model.dec = SparseNeighborDecoder(64)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"query vocabulary ({len(vocab) - 1} tokens):")
    print("  " + " ".join(sorted(v for v in vocab if v != "<pad>")))

    z = model.songs(feats, edge_index)
    queries = sys.argv[1:] or DEFAULT_QUERIES
    for q in queries:
        words = re.findall(r"[a-z]+", q.lower())
        kept = [w for w in words if w in tok2id]
        dropped = [w for w in words if w not in tok2id]
        print(f"\n=== {q!r}")
        print(f"    kept: {kept}   dropped: {dropped}")
        if not kept:
            print("    -> no known tokens, cannot condition")
            continue
        tokens = torch.zeros(1, 6, dtype=torch.long)
        ids = [tok2id[t] for t in kept][:6]
        tokens[0, :len(ids)] = torch.tensor(ids)
        tok_emb, tok_mask, pooled = model.query(tokens)
        with torch.no_grad():
            seq = model.dec.generate(z, tok_emb, tok_mask, pooled, nbrs, length=SEQ_LEN)
        for i, s in enumerate(seq):
            r = songs.iloc[s]
            print(f"    {i+1:>2}  {r['artists'].split(';')[0][:22]:<22} - {r['track_name'][:36]:<36} "
                  f"[{r['track_genre']}] en={r['energy']:.2f} va={r['valence']:.2f} ac={r['acousticness']:.2f}")


if __name__ == "__main__":
    main()
