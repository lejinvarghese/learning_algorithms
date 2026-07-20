# Experiment log

Query-conditioned playlist generation. Real data: 11,339 songs (Zenodo
#nowplaying playlists x HF audio features), test = held-out playlists,
random baseline hit@10 = 0.09%.

| # | Experiment | Data | hit@10 | Notes |
|---|------------|------|--------|-------|
| 0 | Learned token vocab (195 tokens), graph from all pairs | 12,820 pairs | 42.6% | Inflated: test co-listen edges leaked into the graph; easy test subset (vocab-named playlists only) |
| 1 | Frozen ModernBERT queries (gte-modernbert-base), train-only graph | 26,150 pairs | 33.5% | Open vocabulary: zero-shot wins ("melancholic rainy sunday" works); lexical anchors lost ("christmas" smoothed away) |
| 2 | Hybrid: dense + learned exact-match lexical tokens (180 words) | 26,150 pairs | 35.6% | "christmas" anchor recovered (all-Christmas playlist). Unmet-intent rate grows 1.4%→8.6% over training — attention concentrates on dominant facets, motivating exp 3 |
| 3 | + attention-coverage loss (w=1.0) and MMR decoding | 26,150 pairs | 35.7% | Unmet-intent rate 8.6% → 0.0% at zero accuracy cost. "heavy metal love songs" now surfaces GnR/Aerosmith ballads. MMR needs top-k-std scaling (raw logit scale destroys coherence); with it, "Love Hurts" appears while staying in-genre |
| 4 | Intent-dependent arcs, synthetic true-order data, unseen intent combos | 3,000 pairs | 53.1% / 52.9% | Plain GRU already learns intent→arc mapping and generalizes it: energy-trajectory corr +0.749 on fully held-out combos; phase-conditioned decoder +0.797 (modest gain). Unseen "punk night" generates a declining energy walk |
| 5 | Exp-3 recipe at hidden 128, 15 epochs, MPS | 26,150 pairs | 39.7% | Plateaus ~39.5% from epoch 9; unmet intents 0.0% throughout. MPS 100ms/step vs 126ms CPU at this size. Best qualitative results: vietnam query finds Dylan "Blowin' in the Wind" + CCR (real 70s protest material); punk-christmas finds Jimmy Eat World "Christmas Card" |
| 6 | Union graph: adjacency (bias 1.0) + window-4 co-membership edges (log-count weights), 306k edges | 26,150 pairs | 38.8% ✗ | NEGATIVE. Underperforms adjacency-only (39.5% @ same epochs) despite 3x edges. Weighted co-bias gives partial credit everywhere, blurring the sharp adjacency signal the next-song metric rewards. Failure analysis motivating it (73% w/ edge vs 5% w/o) points to content generalization, not softer edges |
