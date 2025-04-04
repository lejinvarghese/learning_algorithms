import click
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
import pandas as pd

from multiprocessing import cpu_count
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

IMAGE_DATASET = "EmbeddingStudio/amazon-products-with-images"
IMAGE_COLS = ["Product Name", "Raw Image"]
TEXT_DATASET = "tasksource/esci"
TEXT_COLS = ["query", "product_title", "esci_label"]
N_PROCESS = cpu_count() - 2


class Preprocessor:
    def __init__(
        self,
        text_model_name,
        vision_model_name,
        sample_size=None,
        batch_size=16,
        num_workers=N_PROCESS,
        shuffle=True,
    ):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_processor = AutoTokenizer.from_pretrained(text_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

    def __download_dataset(self, sample_size):
        images = self.__download_images()
        texts = self.__download_texts()
        data = pd.merge(texts, images, on="product_title", how="inner")
        click.secho(f"Dataset Size: {data.shape}", fg="green")
        if sample_size:
            data = data.sample(sample_size)
            click.secho(f"Sample Dataset Size: {data.shape}", fg="green")
        return Dataset.from_pandas(data)

    def __download_images(self):
        splits = ["train", "eval", "test"]
        images = []
        for s in splits:
            images.append(
                load_dataset(
                    IMAGE_DATASET, columns=IMAGE_COLS, num_proc=N_PROCESS, split=s
                ).to_pandas()
            )
        images = pd.concat(images)
        images.rename(
            columns={"Product Name": "product_title", "Raw Image": "product_image"},
            inplace=True,
        )
        images.dropna(inplace=True)
        images["product_image"] = images["product_image"].apply(lambda x: x["bytes"])
        images["product_title"] = images["product_title"].apply(
            lambda x: str.lower(str.strip(x))
        )
        images.drop_duplicates(subset=["product_title"], inplace=True)
        return images

    def __download_texts(self):
        texts = load_dataset(
            TEXT_DATASET, columns=TEXT_COLS, num_proc=N_PROCESS, split="train"
        ).to_pandas()
        texts = texts[texts["esci_label"].isin(["Exact"])]
        texts["product_title"] = texts["product_title"].apply(
            lambda x: str.lower(str.strip(x))
        )
        return texts

    def collate_fn(self, batch):
        collated_batch = default_data_collator(batch)
        return {key: val.to(self.device) for key, val in collated_batch.items()}

    def encode(self, example):
        anchor = self.text_processor(
            example["query"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        doc_texts = self.text_processor(
            example["product_title"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )
        doc_images = self.image_processor(
            Image.open(BytesIO(example["product_image"])), return_tensors="pt"
        )
        return {
            "anchor_input_ids": anchor["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
            "doc_text_input_ids": doc_texts["input_ids"].squeeze(0),
            "doc_text_attention_mask": doc_texts["attention_mask"].squeeze(0),
            "doc_vision_pixel_values": doc_images["pixel_values"].squeeze(0),
        }

    def run(self):
        dataset = self.__download_dataset(self.sample_size)
        dataset = dataset.map(self.encode, num_proc=self.num_workers)
        dataset.set_format(
            type="torch",
            columns=[
                "anchor_input_ids",
                "anchor_attention_mask",
                "doc_text_input_ids",
                "doc_text_attention_mask",
                "doc_vision_pixel_values",
            ],
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
