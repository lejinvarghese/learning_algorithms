"""Free-text queries against the trained playlist model.

Usage:
  .venv/bin/python infer_text.py "chill acoustic morning" "metal love songs"
  .venv/bin/python infer_text.py                 # interactive REPL
  .venv/bin/python infer_text.py --ckpt data/model_content.pt --mmr 0.5 ...

Without --ckpt, the most recently trained data/model_*.pt is used.
"""

import argparse
import glob
import json
import os
import sys
from rich import print

import torch

from train_real import SEQ_LEN, build_graph, load_data
from train_text import (MAX_LEX, MAX_WORDS, TextPlaylistModel,
                        build_title_cache, content_features, find_anchors,
                        get_encoder, words_of)

DEFAULT_QUERIES = [
    "punk songs for a christmas dinner",
    "country ballads praising the sensual 70s with anti vietnam war sentiment",
    "heavy metal love songs",
    "melancholic rainy sunday introspection",
]


def latest_ckpt():
    paths = glob.glob("data/model_*.pt")
    if not paths:
        sys.exit("no data/model_*.pt found — train one first")
    return max(paths, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mmr", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=0.7,
                    help="sampling temperature; 0 = deterministic greedy")
    ap.add_argument("--topk", type=int, default=20,
                    help="sample among the top-k candidates per step")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the RNG for reproducible sampling")
    ap.add_argument("--max-artist", type=int, default=2, dest="max_artist",
                    help="max songs per artist in a playlist (0 = unlimited)")
    ap.add_argument("--genre-strength", type=float, default=2.5, dest="genre_strength",
                    help="multiplier on the learned genre-match bias (needs --genrebias ckpt). "
                         "Trained value (1.0) is too weak to separate thin/rare-artist genre "
                         "intersections (e.g. rock vs pop Christmas); empirically 2.5 "
                         "differentiates without abandoning the other query facets, 5+ overcorrects.")
    ap.add_argument("queries", nargs="*")
    args = ap.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ckpt_path = args.ckpt or latest_ckpt()
    print(f"model: {ckpt_path}", file=sys.stderr)

    songs, _, feats, _, _ = load_data()
    pairs = [json.loads(l) for l in open("data/playlists_all.jsonl")]
    pairs = [p for p in pairs if len(p["track_ids"]) >= SEQ_LEN]
    edge_index, nbrs = build_graph(pairs, len(songs))

    ckpt = torch.load(ckpt_path)
    lex_vocab = ckpt.get("lex_vocab", [])
    lexmap = ckpt.get("lexmap") or (
        {"words": {w: i + 1 for i, w in enumerate(lex_vocab)}, "phrases": {}}
        if lex_vocab else {})
    fa = songs["artists"].fillna("").str.split(";").str[0].str.lower()
    a2i = {a: i for i, a in enumerate(sorted(set(fa)))}
    all_artist_ids = torch.tensor([a2i[a] for a in fa])
    artist_ids = None
    if ckpt.get("content"):
        artist_ids = all_artist_ids
        feats = torch.cat([feats, content_features(songs), build_title_cache(songs)], 1)
    lex_init = (torch.zeros(len(lex_vocab) + 1, ckpt["emb_dim"])
                if ckpt.get("lexv2") else None)
    model = TextPlaylistModel(feats.size(1), ckpt["emb_dim"], ckpt["hidden"],
                              len(lex_vocab), ckpt.get("n_artists", 0),
                              ckpt.get("layers", 0), lex_init,
                              ckpt.get("steer", False), ckpt.get("genrebias", False),
                              ckpt.get("gat", False))
    model.load_state_dict(ckpt["model"])
    model.eval()
    enc = get_encoder()
    song_genre_ids, n_genres = None, lexmap.get("n_genres", 0)
    if ckpt.get("genrebias"):
        genres = [v.split("genre:", 1)[1] for v in lex_vocab if v.startswith("genre:")]
        g2idx = {g: i for i, g in enumerate(genres)}
        song_genre_ids = torch.tensor(
            [g2idx.get(g, 0) for g in songs["track_genre"].fillna(genres[0])])
    with torch.no_grad():
        z = model.song_z(feats, artist_ids, edge_index)

    def run_query(q):
        ws = words_of(q)[:MAX_WORDS]
        vecs = torch.tensor(enc.encode([q] + ws, normalize_embeddings=True))
        emb = torch.zeros(1, 1 + MAX_WORDS, ckpt["emb_dim"])
        mask = torch.zeros(1, 1 + MAX_WORDS, dtype=torch.bool)
        emb[0, :len(vecs)] = vecs
        mask[0, :len(vecs)] = True
        lex = torch.zeros(1, MAX_LEX, dtype=torch.long)
        hit_ids = find_anchors(q, lexmap)
        lex_hits = [lex_vocab[i - 1] for i in hit_ids]
        if hit_ids:
            lex[0, :len(hit_ids)] = torch.tensor(hit_ids)
        qgenre = None
        ag = lexmap.get("anchor_genre")
        if ag and n_genres:
            qgenre = torch.zeros(1, n_genres)
            for aid in hit_ids:
                if aid in ag:
                    qgenre[0, ag[aid]] = 1.0
        tok, tmask, pooled = model.query(emb, mask, lex)
        with torch.no_grad():
            seq = model.dec.generate(z, tok, tmask, pooled, nbrs,
                                     length=SEQ_LEN, temp=args.temp,
                                     topk=args.topk, mmr=args.mmr,
                                     artists=all_artist_ids,
                                     max_artist=args.max_artist,
                                     song_genre_ids=song_genre_ids, qgenre=qgenre,
                                     genre_strength=args.genre_strength)
        anchors = f"   lexical anchors: {lex_hits}" if lex_vocab else ""
        print(f"\n=== {q!r}{anchors}")
        for i, s in enumerate(seq):
            r = songs.iloc[s]
            print(f"[green]{i+1:>2}  {str.lower(r['artists'].split(';')[0])[:22]:<22} - "
                  f"[cyan]{str.lower(r['track_name'])[:36]:<36} [{r['track_genre']}] [/cyan]"
                  f"[cyan]en={r['energy']:.2f} va={r['valence']:.2f} ac={r['acousticness']:.2f}[/cyan]")

    if args.queries:
        for q in args.queries:
            run_query(q)
    elif sys.stdin.isatty():
        from rich.console import Console
        console = Console()
        print("[yellow]interactive mode — type a query, empty line or Ctrl-D to quit[/yellow]")
        while True:
            try:
                q = console.input("\n[yellow]query> [/yellow]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            run_query(q)
    else:
        for q in DEFAULT_QUERIES:
            run_query(q)


if __name__ == "__main__":
    main()
