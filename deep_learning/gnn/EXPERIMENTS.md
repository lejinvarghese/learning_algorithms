# Experiment log

Query-conditioned playlist generation. Real data: 11,339 songs (Zenodo
#nowplaying playlists x HF audio features), test = held-out playlists,
random baseline hit@10 = 0.09%.

| # | Experiment | Data | hit@10 | Notes |
|---|------------|------|--------|-------|
| 0 | Learned token vocab (195 tokens), graph from all pairs | 12,820 pairs | 42.6% | Inflated: test co-listen edges leaked into the graph; easy test subset (vocab-named playlists only) |
| 1 | Frozen ModernBERT queries (gte-modernbert-base), train-only graph | 26,150 pairs | 33.5% | Open vocabulary: zero-shot wins ("melancholic rainy sunday" works); lexical anchors lost ("christmas" smoothed away) |
| 2 | Hybrid: dense + learned exact-match lexical tokens (180 words) | 26,150 pairs | 35.6% | "christmas" anchor recovered (all-Christmas playlist). Unmet-intent rate grows 1.4%→8.6% over training — attention concentrates on dominant facets, motivating exp 3 |
