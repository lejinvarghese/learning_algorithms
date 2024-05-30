import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import (
    MultipleNegativesRankingLoss,
    CachedGISTEmbedLoss,
    TripletLoss,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from multiprocessing import cpu_count
from transformers.integrations import TensorBoardCallback

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
esci = load_dataset(DATASET_NAME, num_proc=N_PROCS, columns=cols)
esci_train_df = esci["train"].to_pandas()
esci_valid_df = esci["test"].to_pandas()
print(esci_train_df.shape, esci_valid_df.shape)


prompts = {"query": "search_query: ", "document": "search_document:"}


def build_triplets(df, prompts, sample_frac=0.1):
    # df = df[df["esci_label"].isin(["E", "I"])]
    df["label"] = np.where(df["esci_label"].isin(["Exact"]), 1, 0)
    df["document"] = (
        df["product_title"]
        # + ", description: "
        # + df["product_description"]
        + ", "
        + df["product_brand"]
        + ", "
        + df["product_color"]
    )
    pos = df[df["label"] == 1]
    pos_nec = pos[["query", "document"]]
    pos_nec.loc[:, "query"] = pos_nec["query"].apply(
        lambda x: f"{prompts.get('query', '')}{x}"
    )
    pos_nec.loc[:, "document"] = pos_nec["document"].apply(
        lambda x: f"{prompts.get('document', '')}{x}"
    )
    pos_nec.columns = ["anchor", "positive"]

    neg = df[df["label"] == 0]
    neg_nec = neg[["query", "document"]]
    neg_nec.loc[:, "query"] = neg_nec["query"].apply(
        lambda x: f"{prompts.get('query', '')}{x}"
    )
    neg_nec.loc[:, "document"] = neg_nec["document"].apply(
        lambda x: f"{prompts.get('document', '')}{x}"
    )
    neg_nec.columns = ["anchor", "negative"]

    triplets = pos_nec.merge(neg_nec, on="anchor", how="inner").sample(
        frac=sample_frac, random_state=RANDOM_STATE
    )
    return Dataset.from_pandas(triplets, preserve_index=False)


def main():
    training_dataset = build_triplets(esci_train_df, prompts=prompts, sample_frac=0.25)
    print(f"Finished sampling triplets:{training_dataset.shape}")

    # 1. Load a model to finetune
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )

    # 3. Define a loss function
    guide = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)
    loss = CachedGISTEmbedLoss(model, guide=guide, mini_batch_size=4)
    # loss = TripletLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir="models/nomic-embed-text-esci",
        seed=RANDOM_STATE,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine_with_restarts",
        fp16=True,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # eval_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
        logging_steps=100,
        run_name="nomic-embed-text-esci",
        disable_tqdm=False,
    )

    # 4. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=training_dataset,
        loss=loss,
        callbacks=[TensorBoardCallback()],
        # eval_dataset=validation_dataset,
        # evaluator=dev_evaluator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
