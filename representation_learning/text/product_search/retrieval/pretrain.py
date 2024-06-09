import os
import logging
import click
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import (
    GISTEmbedLoss,
    MatryoshkaLoss,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    SequentialEvaluator,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
    InformationRetrievalEvaluator,
)
from transformers import EarlyStoppingCallback

from data import DataLoader

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["LOCAL_RANK"] = "0"
k = 10


@click.command()
@click.option("--n_samples", default=10_000, help="Number of samples to use for training")
def main(n_samples):
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    dataloader = DataLoader()
    train_positives_dataset, valid_positives_dataset = (
        dataloader.generate_positives(split="train", n_samples=n_samples),
        dataloader.generate_positives(split="test", n_samples=n_samples // 100),
    )
    valid_pairs_dataset = dataloader.generate_pairs(split="test", n_samples=n_samples // 100)

    queries, corpus, qrels = dataloader.generate_ir_datasets()

    train_dataset = {"positives": train_positives_dataset}
    valid_dataset = {"positives": valid_positives_dataset}
    guide = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-6-v2", trust_remote_code=True)
    losses = {
        "positives": GISTEmbedLoss(model, guide=guide),
    }
    losses = {k: MatryoshkaLoss(model, v, [768, 512, 256, 128, 64]) for k, v in losses.items()}

    similarity_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=valid_pairs_dataset["anchor"],
        sentences2=valid_pairs_dataset["document"],
        scores=valid_pairs_dataset["score"],
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
    seq_evaluator = SequentialEvaluator([similarity_evaluator, ir_evaluator])

    args = SentenceTransformerTrainingArguments(
        output_dir="models/nomic-embed-text-pretrain-esci",
        run_name="nomic-embed-text-pretrain-esci",
        seed=42,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=4,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.02,
        learning_rate=5e-7,
        lr_scheduler_type="polynomial",
        # lr_scheduler_kwargs={"num_cycles": 1},
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=20,
        logging_steps=100,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        dataloader_drop_last=True,
        do_eval=True,
        eval_delay=0,
        eval_steps=100,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_cosine_ndcg@{k}",
        gradient_checkpointing=True,
        disable_tqdm=False,
    )

    # 4. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        evaluator=seq_evaluator,
        loss=losses,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001),
        ],
        compute_metrics=seq_evaluator,
        args=args,
    )
    trainer.train()
    seq_evaluator(model)


if __name__ == "__main__":
    main()
