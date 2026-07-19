# playlist-gnn

Query-conditioned playlist generation over a directed song graph (PyTorch
Geometric). Songs are nodes with audio features; playlists provide
co-listen edges; free-text queries condition an autoregressive decoder
that walks the graph via cross-attention over query tokens.

See `EXPERIMENTS.md` for the experiment ladder and results.

## Files

- `playlist_gnn.py` — synthetic prototype: direction-aware GraphSAGE song
  encoder, intent-token cross-attention GRU decoder, graph-edge logit bias.
- `prepare_real.py` — downloads/joins real data: Zenodo #nowplaying Spotify
  playlists x HF 114k-track audio features; mines playlist names as queries.
- `train_real.py` — exp 0: learned 195-token query vocab on real data.
- `train_text.py` — exps 1-3, 5: frozen ModernBERT (gte-modernbert-base)
  query embeddings; `--lexical` adds exact-match anchor tokens; `--coverage`
  adds the attention-coverage loss; `--device mps` for Apple GPU.
- `infer_text.py` — free-text inference (`--mmr` for diversified decoding).
- `exp_arcs.py` — exp 4: intent-dependent sequential arcs on synthetic data
  with true order; plain GRU vs phase-conditioned decoder on unseen combos.
- `bench_device.py` — cpu vs mps step-time benchmark.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_real.py      # downloads ~200MB, builds data/
.venv/bin/python train_text.py --lexical --coverage 1.0
.venv/bin/python infer_text.py "melancholic rainy sunday introspection"
```

Data caveat: this playlist dump alphabetizes track order, so real-data
edges are co-listening, not true transitions (hence exp 4 is synthetic).
