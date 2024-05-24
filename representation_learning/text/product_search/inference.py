import click
import numpy as np
import logging
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from constants import DIRECTORY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cols = [
    "query_id",
    "product_id",
    "query",
    "product_title",
    "product_brand",
    "product_color",
    "gain",
]


class Predictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(
            self.device
        )
        logger.info(f"Loaded model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def predict(self, queries: str, products: str, batch_size: int = 32):
        n_examples = len(queries)
        scores = np.zeros(n_examples)

        with torch.no_grad():
            for i in tqdm(range(0, n_examples, batch_size)):
                j = min(i + batch_size, n_examples)
                features = self.tokenizer(
                    queries[i:j],
                    products[i:j],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                scores[i:j] = np.squeeze(
                    self.model(**features).logits.cpu().detach().numpy()
                )
                i = j
        return scores


@click.command()
@click.option("--sample", type=bool, default=False, help="Sample queries.")
def main(sample):

    test_queries = [
        "champion reverse weave",
        "chacos women sandals wide width 9",
        "girls surf backpack",
    ]
    df = pd.read_parquet(f"{DIRECTORY}/test.parquet")[cols]
    if sample:
        df = df[df["query"].isin(test_queries)]
    queries = df["query"].to_list()
    products = df["product_title"].to_list()

    model_path = f"{DIRECTORY}/models/cross-encoder/ms-marco-MiniLM-L-12-v2"
    pr = Predictor(model_path)
    df["score"] = pr.predict(queries, products)
    df.sort_values(by=["query_id", "score"], ascending=False, inplace=True)
    logger.info(df[cols[2:] + ["score"]].head(10))
    df.to_parquet(f"{DIRECTORY}/test_scored.parquet")


if __name__ == "__main__":
    main()
