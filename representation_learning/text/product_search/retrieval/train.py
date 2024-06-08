import os
import logging
import click
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import (
    CachedGISTEmbedLoss,
    MultipleNegativesRankingLoss,
    CachedMultipleNegativesRankingLoss,
    TripletLoss,
    TripletDistanceMetric,
    AnglELoss,
    MatryoshkaLoss,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    TripletEvaluator,
    SequentialEvaluator,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)
from transformers import EarlyStoppingCallback

from data import DataLoader

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LOCAL_RANK"] = "0"


@click.command()
@click.option("--n_samples", default=10_000, help="Number of samples to use for training")
def main(n_samples):
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    dataloader = DataLoader()
    train_triplets_dataset, valid_triplets_dataset = (
        dataloader.generate_triplets(split="train", n_samples=n_samples),
        dataloader.generate_triplets(split="test", n_samples=n_samples // 100),
    )

    train_pairs_dataset, valid_pairs_dataset = (
        dataloader.generate_pairs(split="train", n_samples=n_samples),
        dataloader.generate_pairs(split="test", n_samples=n_samples // 100),
    )

    # guide = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-6-v2", trust_remote_code=True)
    # loss = CachedGISTEmbedLoss(model, guide=guide, mini_batch_size=4)
    # loss = CachedMultipleNegativesRankingLoss(model)

    # pairs_loss = CoSENTLoss(model)
    train_dataset = {"triplets": train_triplets_dataset, "pairs": train_pairs_dataset}
    valid_dataset = {"triplets": valid_triplets_dataset, "pairs": valid_pairs_dataset}
    losses = {
        "triplets": TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.5),
        "pairs": AnglELoss(model),
    }
    losses = {k: MatryoshkaLoss(model, v, [768, 512, 256, 128, 64]) for k, v in losses.items()}

    triplets_evaluator = TripletEvaluator(
        anchors=valid_triplets_dataset["anchor"],
        positives=valid_triplets_dataset["positive"],
        negatives=valid_triplets_dataset["negative"],
        main_distance_function=SimilarityFunction.COSINE,
    )
    pairs_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=valid_pairs_dataset["query"],
        sentences2=valid_pairs_dataset["document"],
        scores=valid_pairs_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
    )
    seq_evaluator = SequentialEvaluator([triplets_evaluator, pairs_evaluator])

    args = SentenceTransformerTrainingArguments(
        output_dir="models/nomic-embed-text-esci",
        run_name="nomic-embed-text-esci",
        seed=42,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        # auto_find_batch_size=True,
        gradient_accumulation_steps=2,
        warmup_ratio=0.01,
        learning_rate=1e-6,
        lr_scheduler_type="cosine_with_restarts",
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
        logging_steps=100,
        eval_steps=1000,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        disable_tqdm=False,
        evaluation_strategy="steps",
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        metric_for_best_model="cosine_accuracy",
        lr_scheduler_kwargs={"num_cycles": 1},
        gradient_checkpointing=True,
    )

    # 4. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        evaluator=seq_evaluator,
        loss=losses,
        callbacks=[
            # EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001),
        ],
        compute_metrics=seq_evaluator,
        args=args,
    )
    trainer.train()
    seq_evaluator(model)


if __name__ == "__main__":
    main()
