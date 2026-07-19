# playlist-gnn

Prototype: query-conditioned playlist generation over a directed song-transition
graph, built with PyTorch Geometric.

- Songs = nodes (audio-ish features + genre); consecutive plays in simulated
  listening sessions = sparse directed edges.
- `SongEncoder`: direction-aware GraphSAGE (separate in/out aggregation) —
  inductive, so cold-start songs only need features.
- `QueryEncoder`: intent tokens ("chill", "punk", "sunny", ...) kept as
  token-level embeddings, not just a pooled vector.
- `PlaylistDecoder`: GRU that walks the graph autoregressively, cross-attending
  over query tokens at every step, with a learned logit bias toward graph
  out-neighbors of the current song.
- Training: teacher-forced next-song cross-entropy on (query, playlist) pairs.

Run:

```
.venv/bin/python playlist_gnn.py --query "chill punk rock vibes for a sunny morning"
```

To productionize, swap the toy intent vocabulary for a frozen text encoder
(e.g. sentence-transformers or CLAP) feeding the same cross-attention.
