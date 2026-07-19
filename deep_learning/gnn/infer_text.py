"""Free-text queries against the ModernBERT-conditioned playlist model.

Usage:  .venv/bin/python infer_text.py [--ckpt data/model_text.pt] [--mmr 0.5] \
            "country ballads with anti war sentiment" ...
With no query args, runs the conflicting-query battery.
"""

import argparse
import json
import re

import torch

from train_real import SEQ_LEN, build_graph, load_data
from train_text import (MAX_LEX, MAX_WORDS, TextPlaylistModel, get_encoder,
                        words_of)

DEFAULT_QUERIES = [
    "punk songs for a christmas dinner",
    "country ballads praising the sensual 70s with anti vietnam war sentiment",
    "sad songs for a summer party",
    "heavy metal love songs",
    "melancholic rainy sunday introspection",
    "energetic latin dance fiesta",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/model_text.pt")
    ap.add_argument("--mmr", type=float, default=0.0)
    ap.add_argument("queries", nargs="*")
    args = ap.parse_args()

    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]
    edge_index, nbrs = build_graph(pairs, len(songs))

    ckpt = torch.load(args.ckpt)
    lex_vocab = ckpt.get("lex_vocab", [])
    lex2id = {w: i + 1 for i, w in enumerate(lex_vocab)}
    model = TextPlaylistModel(feats.size(1), ckpt["emb_dim"], ckpt["hidden"],
                              len(lex_vocab))
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = get_encoder()
    z = model.songs(feats, edge_index)

    for q in (args.queries or DEFAULT_QUERIES):
        ws = words_of(q)[:MAX_WORDS]
        vecs = torch.tensor(enc.encode([q] + ws, normalize_embeddings=True))
        emb = torch.zeros(1, 1 + MAX_WORDS, ckpt["emb_dim"])
        mask = torch.zeros(1, 1 + MAX_WORDS, dtype=torch.bool)
        emb[0, :len(vecs)] = vecs
        mask[0, :len(vecs)] = True
        lex = torch.zeros(1, MAX_LEX, dtype=torch.long)
        lex_hits = [w for w in words_of(q) if w in lex2id][:MAX_LEX]
        lex[0, :len(lex_hits)] = torch.tensor([lex2id[w] for w in lex_hits])
        tok, tmask, pooled = model.query(emb, mask, lex)
        with torch.no_grad():
            seq = model.dec.generate(z, tok, tmask, pooled, nbrs,
                                     length=SEQ_LEN, mmr=args.mmr)
        anchors = f"   lexical anchors: {lex_hits}" if lex_vocab else ""
        print(f"\n=== {q!r}{anchors}")
        for i, s in enumerate(seq):
            r = songs.iloc[s]
            print(f"  {i+1:>2}  {r['artists'].split(';')[0][:22]:<22} - {r['track_name'][:36]:<36} "
                  f"[{r['track_genre']}] en={r['energy']:.2f} va={r['valence']:.2f} ac={r['acousticness']:.2f}")


if __name__ == "__main__":
    main()
