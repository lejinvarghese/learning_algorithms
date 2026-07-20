# External research: playlist continuation & GNN recommenders

Findings from a literature survey (July 2026) relevant to this project's
failure modes (no-edge transitions 5%, rare songs 10%, seed prediction 25%).

## Most relevant published systems

- **Text2Tracks** (Spotify 2025, [arXiv:2503.24193](https://arxiv.org/abs/2503.24193)) —
  prompt→track generative retrieval; our exact domain. Semantic IDs built
  from CF co-occurrence embeddings beat title-string IDs (+48%) and dense
  bi-encoders (+127%); Hits@10 0.270 over 500K tracks, no seeds.
- **RecSys Challenge 2018 analysis** ([arXiv:1810.01520](https://arxiv.org/pdf/1810.01520)) —
  winning MPD solutions = CF retrieval + XGBoost rerank; random playlist
  subsets condition better than sequential prefixes; title tokens can HURT
  seeded playlists; title-only cold start needs a dedicated path.
  Top scores: R-precision 0.224 / NDCG 0.395 over 2.2M tracks.
- **PinSage** ([arXiv:1806.01973](https://arxiv.org/pdf/1806.01973)) — never
  densify edges; define neighborhoods by random-walk (PPR) visit counts
  (+46% vs k-hop) and use PPR-ranked curriculum hard negatives.
  Explains our exp-6 negative result.
- **LightGCN/UltraGCN** ([arXiv:2002.02126](https://arxiv.org/pdf/2002.02126)) —
  depth/nonlinearity do NOT help implicit-feedback ranking; propagation
  structure and loss/negatives do. Tempers expectations for deep GNNs.
- **TIGER** ([arXiv:2305.05065](https://arxiv.org/pdf/2305.05065)) + YouTube
  semantic-ID ranking ([arXiv:2306.08121](https://arxiv.org/pdf/2306.08121)) —
  RQ-VAE semantic IDs let rare items inherit statistics from popular
  neighbors; help even as plain input features.
- **LARP** ([arXiv:2406.14333](https://arxiv.org/abs/2406.14333)) — staged
  contrastive pretraining aligning content-text embeddings with the CF
  space; the published recipe for cold-start playlist continuation.
- **Loss literature** (gSASRec; sampled-softmax TOIS 2024;
  [logQ correction](https://arxiv.org/html/2507.09331v2)) — full softmax CE
  with popularity (logQ) correction beats BPR/BCE and approx-NDCG
  surrogates for top-K sequential rec.
- **Spotify production**: RL playlist sequencing vs a user simulator
  ([arXiv:2310.09123](https://arxiv.org/abs/2310.09123)); algotorial
  playlists (editor pool + ML ordering); AI Playlist = LLM agent + tools,
  DPO on skip/save signals. Deezer ships represent-then-aggregate APC
  ([arXiv:2304.09061](https://arxiv.org/abs/2304.09061), public eval harness).

## Resulting experiment queue (post exp 7)

1. logQ popularity-corrected softmax (before listwise comparison)
2. Random-context conditioning (shuffle training order; our order is
   alphabetical noise anyway)
3. PPR-weighted neighbor bias + curriculum hard negatives (PinSage recipe)
4. Deep residual GNN --layers 4 (cheap; literature predicts small gains)
5. Semantic-ID output vocabulary (RQ-VAE over CF embeddings, trie decoding)
6. Eval additions: R-precision with artist partial credit,
   popularity-stratified reporting
