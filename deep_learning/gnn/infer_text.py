"""Free-text queries against the ModernBERT-conditioned playlist model.

Usage:  .venv/bin/python infer_text.py "country ballads with anti war sentiment" ...
With no args, runs the conflicting-query battery.
"""

import json
import re
import sys

import torch

from train_real import SEQ_LEN, build_graph, load_data
from train_text import MAX_WORDS, TextPlaylistModel, get_encoder

DEFAULT_QUERIES = [
    "punk songs for a christmas dinner",
    "country ballads praising the sensual 70s with anti vietnam war sentiment",
    "sad songs for a summer party",
    "heavy metal love songs",
    "melancholic rainy sunday introspection",
    "energetic latin dance fiesta",
]


def main():
    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]
    edge_index, nbrs = build_graph(pairs, len(songs))

    ckpt = torch.load("data/model_text.pt")
    model = TextPlaylistModel(feats.size(1), ckpt["emb_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = get_encoder()
    z = model.songs(feats, edge_index)

    for q in (sys.argv[1:] or DEFAULT_QUERIES):
        words = [w for w in re.findall(r"[a-z0-9]+", q.lower()) if len(w) >= 2]
        words = list(dict.fromkeys(words))[:MAX_WORDS]
        vecs = torch.tensor(enc.encode([q] + words, normalize_embeddings=True))
        emb = torch.zeros(1, 1 + MAX_WORDS, ckpt["emb_dim"])
        mask = torch.zeros(1, 1 + MAX_WORDS, dtype=torch.bool)
        emb[0, :len(vecs)] = vecs
        mask[0, :len(vecs)] = True
        tok, tmask, pooled = model.query(emb, mask)
        with torch.no_grad():
            seq = model.dec.generate(z, tok, tmask, pooled, nbrs, length=SEQ_LEN)
        print(f"\n=== {q!r}")
        for i, s in enumerate(seq):
            r = songs.iloc[s]
            print(f"  {i+1:>2}  {r['artists'].split(';')[0][:22]:<22} - {r['track_name'][:36]:<36} "
                  f"[{r['track_genre']}] en={r['energy']:.2f} va={r['valence']:.2f} ac={r['acousticness']:.2f}")


if __name__ == "__main__":
    main()
