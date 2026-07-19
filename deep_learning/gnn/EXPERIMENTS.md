# Experiment log

Query-conditioned playlist generation. Real data: 11,339 songs (Zenodo
#nowplaying playlists x HF audio features), test = held-out playlists,
random baseline hit@10 = 0.09%.

| # | Experiment | Data | hit@10 | Notes |
|---|------------|------|--------|-------|
| 0 | Learned token vocab (195 tokens), graph from all pairs | 12,820 pairs | 42.6% | Inflated: test co-listen edges leaked into the graph; easy test subset (vocab-named playlists only) |
| 1 | Frozen ModernBERT queries (gte-modernbert-base), train-only graph | 26,150 pairs | 33.5% | Open vocabulary: zero-shot wins ("melancholic rainy sunday" works); lexical anchors lost ("christmas" smoothed away) |
