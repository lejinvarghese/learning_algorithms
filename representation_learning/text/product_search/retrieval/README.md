# Train

```bash
python3.10 -m train --n_samples
python3.10 -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port=12345 train.py
 ```

## Plots

![image](../assets/accuracy_curves.png)


## Baseline Model
| Evaluator                     | Metric                      | Value  |
| ----------------------------- | --------------------------- | ------ |
| TripletEvaluator              | Accuracy Cosine Distance    | 0.7036 |
| TripletEvaluator              | Accuracy Dot Product        | 0.2995 |
| TripletEvaluator              | Accuracy Manhattan Distance | 0.7015 |
| TripletEvaluator              | Accuracy Euclidean Distance | 0.7009 |
| InformationRetrievalEvaluator | Accuracy@10                 | 0.9580 |
| InformationRetrievalEvaluator | Precision@10                | 0.5651 |
| InformationRetrievalEvaluator | Recall@10                   | 0.6862 |
| InformationRetrievalEvaluator | MRR@10                      | 0.8937 |
| InformationRetrievalEvaluator | NDCG@10                     | 0.8029 |
| InformationRetrievalEvaluator | MAP@10                      | 0.7387 |
| EmbeddingSimilarity           | pearson_cosine              | 0.3928 |
| EmbeddingSimilarity           | spearman_cosine             | 0.3822 |
| EmbeddingSimilarity           | pearson_manhattan           | 0.3577 |
| EmbeddingSimilarity           | spearman_manhattan          | 0.3479 |
| EmbeddingSimilarity           | pearson_euclidean           | 0.3590 |
| EmbeddingSimilarity           | spearman_euclidean          | 0.3491 |