import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import (
    MultipleNegativesRankingLoss,
    CachedGISTEmbedLoss,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from multiprocessing import cpu_count
from transformers.integrations import TensorBoardCallback

esci = load_dataset(
    "smangrul/amazon_esci", num_proc=cpu_count() - 2, ignore_verifications=True
)
esci_train_df = esci["train"].to_pandas()
esci_valid_df = esci["validation"].to_pandas()

guide = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)


def build_multi_neg_dataset(df, query_prompt="", doc_prompt=""):
    df = df[df["esci_label"].isin(["E", "I"])]
    df["label"] = np.where(df["esci_label"].isin(["E"]), 1, 0)
    pos = df[df["label"] == 1]
    pos_nec = pos[["query", "product_title"]]
    pos_nec["query"] = pos_nec["query"].apply(lambda x: f"{query_prompt}{x}")
    pos_nec["product_title"] = pos_nec["product_title"].apply(
        lambda x: f"{doc_prompt}{x}"
    )
    pos_nec.columns = ["query", "pos"]

    neg = df[df["label"] == 0]
    neg_nec = neg[["query", "product_title"]]
    neg_nec["query"] = neg_nec["query"].apply(lambda x: f"{query_prompt}{x}")
    neg_nec["product_title"] = neg_nec["product_title"].apply(
        lambda x: f"{doc_prompt}{x}"
    )
    neg_nec.columns = ["query", "neg"]

    posneg = pos_nec.merge(neg_nec, on="query", how="inner").sample(
        frac=0.5, random_state=42
    )

    return Dataset.from_pandas(posneg, preserve_index=False)


training_dataset, validation_dataset = build_multi_neg_dataset(
    esci_train_df, query_prompt="search_query: ", doc_prompt="search_document: "
), build_multi_neg_dataset(
    esci_valid_df, query_prompt="search_query: ", doc_prompt="search_document: "
)
print("finished building datasets")

# 1. Load a model to finetune
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# 3. Define a loss function
loss = CachedGISTEmbedLoss(model, guide=guide, mini_batch_size=4)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/nomic-embed-text-esci",
    # Optional training parameters:
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if GPU can't handle FP16
    bf16=False,  # Set to True if GPU supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
    # Optional tracking/debugging parameters:
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    logging_steps=100,
    run_name="nomic-embed-text-esci",  # Used in W&B if `wandb` is installed
)

dev_evaluator = TripletEvaluator(
    anchors=validation_dataset["query"],
    positives=validation_dataset["pos"],
    negatives=validation_dataset["neg"],
    name="esci-dev",
)
dev_evaluator(model)
# 4. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    loss=loss,
    callbacks=[TensorBoardCallback()],
    evaluator=dev_evaluator,
)
trainer.train()

# test_evaluator = TripletEvaluator(
#     anchors=test_dataset["anchor"],
#     positives=test_dataset["positive"],
#     negatives=test_dataset["negative"],
#     name="all-nli-test",
# )
# test_evaluator(model)
