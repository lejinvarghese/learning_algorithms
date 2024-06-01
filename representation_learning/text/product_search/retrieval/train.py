import os
import click
import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import (
    CachedGISTEmbedLoss,
    CachedMultipleNegativesRankingLoss,
    TripletLoss,
    CoSENTLoss,
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
from multiprocessing import cpu_count
from transformers.integrations import TensorBoardCallback

import logging

logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["LOCAL_RANK"] = "0"

RANDOM_STATE = 42
DATASET_NAME = "tasksource/esci"
N_PROCS = cpu_count() - 2
cols = [
    "product_title",
    "product_description",
    "product_brand",
    "product_color",
    "esci_label",
    "query",
]
esci = load_dataset("tasksource/esci", num_proc=N_PROCS, columns=cols)
esci_train_df = esci["train"].to_pandas()
esci_valid_df = esci["test"].to_pandas()
click.secho(f"Initial dataset sizes: {esci_train_df.shape, esci_valid_df.shape}", fg="green")

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
prompts = {"query": "search_query: ", "document": "search_document: "}


def build_dataset(df, prompts, n_samples=0.1, dataset_type="triplets"):
    # df = df[df["esci_label"].isin(["E", "I"])]
    df = df.dropna()

    conditions = [
        df["esci_label"].isin(["Exact"]),
        df["esci_label"].isin(["Substitute"]),
        df["esci_label"].isin(["Complement"]),
    ]
    grades = [1.0, 0.2, 0.01]
    df["grade"] = np.select(conditions, grades, default=0)

    df["document"] = (
        df["product_title"]
        # + ",  "
        # + df["product_description"]
        + ", "
        + df["product_brand"]
        + ", "
        + df["product_color"]
    )
    pairs = df[["query", "document", "grade"]].copy()
    pos = df[df["grade"] == 1]
    pos_nec = pos[["query", "document"]]
    pos_nec.loc[:, "query"] = pos_nec["query"].apply(lambda x: f"{prompts.get('query', '')}{x}")
    pos_nec.loc[:, "document"] = pos_nec["document"].apply(lambda x: f"{prompts.get('document', '')}{x}")
    pos_nec.columns = ["anchor", "positive"]

    neg = df[df["grade"] < 1]
    neg_nec = neg[["query", "document"]]
    neg_nec.loc[:, "query"] = neg_nec["query"].apply(lambda x: f"{prompts.get('query', '')}{x}")
    neg_nec.loc[:, "document"] = neg_nec["document"].apply(lambda x: f"{prompts.get('document', '')}{x}")
    neg_nec.columns = ["anchor", "negative"]

    triplets = pos_nec.merge(neg_nec, on="anchor", how="inner").sample(n_samples, random_state=RANDOM_STATE)[
        ["anchor", "positive", "negative"]
    ]
    click.secho(triplets.head())
    if dataset_type == "pairs":
        click.secho(pairs.head())
        pairs = pairs.sample(n_samples, random_state=RANDOM_STATE)
        pairs.columns = ["sentence1", "sentence2", "score"]
        return Dataset.from_pandas(pairs, preserve_index=False)
    else:
        return Dataset.from_pandas(triplets, preserve_index=False)


@click.command()
@click.option("--n_samples", default=10_000, help="Number of samples to use for training")
def main(n_samples):
    train_triplets_dataset, valid_triplets_dataset = (
        build_dataset(esci_train_df, prompts=prompts, n_samples=n_samples),
        build_dataset(esci_valid_df, prompts=prompts, n_samples=n_samples // 100),
    )
    click.secho(
        f"Finished sampling triplets:{train_triplets_dataset.shape}, {valid_triplets_dataset.shape}",
        fg="yellow",
    )

    train_pairs_dataset, valid_pairs_dataset = (
        build_dataset(esci_train_df, prompts=prompts, n_samples=n_samples, dataset_type="pairs"),
        build_dataset(
            esci_valid_df,
            prompts=prompts,
            n_samples=n_samples // 100,
            dataset_type="pairs",
        ),
    )
    click.secho(
        f"Finished sampling pairs:{train_pairs_dataset.shape}, {valid_pairs_dataset.shape}",
        fg="yellow",
    )

    guide = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-6-v2", trust_remote_code=True)
    loss = CachedGISTEmbedLoss(model, guide=guide, mini_batch_size=4)
    # loss = TripletLoss(model)
    # pairs_loss = CoSENTLoss(model)
    train_dataset = {"triplets": train_triplets_dataset}  # , "pairs": train_pairs_dataset}
    valid_dataset = {"triplets": valid_triplets_dataset}  # , "pairs": valid_pairs_dataset}
    # losses = {"triplets": triplets_loss}  # , "pairs": pairs_loss}

    triplets_evaluator = TripletEvaluator(
        anchors=valid_triplets_dataset["anchor"],
        positives=valid_triplets_dataset["positive"],
        negatives=valid_triplets_dataset["negative"],
        main_distance_function=SimilarityFunction.COSINE,
        name="triplet-esci",
    )
    pairs_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=valid_pairs_dataset["sentence1"],
        sentences2=valid_pairs_dataset["sentence2"],
        scores=valid_pairs_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="pairs-esci",
    )
    seq_evaluator = SequentialEvaluator([triplets_evaluator])  # , pairs_evaluator])

    args = SentenceTransformerTrainingArguments(
        output_dir="models/nomic-embed-text-esci",
        run_name="nomic-embed-text-esci",
        seed=RANDOM_STATE,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # auto_find_batch_size=True,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        learning_rate=1e-5,  # learning_rate=1e-6,
        lr_scheduler_type="cosine_with_restarts",
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=10,
        logging_steps=200,
        eval_steps=1000,
        dataloader_num_workers=4,
        disable_tqdm=False,
        evaluation_strategy="steps",
        dataloader_drop_last=True,
    )

    # 4. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_triplets_dataset,
        loss=loss,
        callbacks=[TensorBoardCallback()],
        eval_dataset=valid_triplets_dataset,
        evaluator=seq_evaluator,
    )
    trainer.train()
    seq_evaluator(model)


if __name__ == "__main__":
    main()
