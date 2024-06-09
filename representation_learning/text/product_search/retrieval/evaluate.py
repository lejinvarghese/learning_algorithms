import click
import logging
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    TripletEvaluator,
    SequentialEvaluator,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
    InformationRetrievalEvaluator,
)
from data import DataLoader

logging.basicConfig(level=logging.INFO)

models = [
    "nomic-ai/nomic-embed-text-v1.5",
]


@click.command()
@click.option("--n_samples", default=10_000, help="Number of samples to use for training")
@click.option("--k", default=10, help="k for information retrieval evaluation")
def main(n_samples, k):
    dataloader = DataLoader()
    queries, corpus, qrels = dataloader.generate_ir_datasets()
    triplets = dataloader.generate_triplets(split="test", n_samples=n_samples)
    pairs = dataloader.generate_pairs(split="test", n_samples=n_samples)

    triplets_evaluator = TripletEvaluator(
        anchors=triplets["anchor"],
        positives=triplets["positive"],
        negatives=triplets["negative"],
        main_distance_function=SimilarityFunction.COSINE,
    )
    similarity_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=pairs["anchor"],
        sentences2=pairs["document"],
        scores=pairs["score"],
        main_similarity=SimilarityFunction.COSINE,
    )
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        show_progress_bar=True,
        corpus_chunk_size=256,
        accuracy_at_k=[k],
        precision_recall_at_k=[k],
        map_at_k=[k],
        main_score_function=SimilarityFunction.COSINE,
    )
    seq_evaluator = SequentialEvaluator([triplets_evaluator, similarity_evaluator, ir_evaluator])

    for m in models:
        model = SentenceTransformer(m, trust_remote_code=True)
        click.secho(seq_evaluator(model))


if __name__ == "__main__":
    main()
