import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from PIL import Image
import requests

from data_loader import Preprocessor


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=768, n_layers=2, dropout=0.2):
        super().__init__()

        blocks = [
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            # nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
        ]
        layers = []
        for _ in range(n_layers):
            layers.extend(blocks)
        layers.extend(
            [
                nn.Linear(output_dim, output_dim),
            ]
        )
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x)
        return x


class ThreeTowerRetrievalModel(nn.Module):
    def __init__(
        self, text_model_name: str, vision_model_name: str, embedding_dim: int
    ):
        super().__init__()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.embedding_dim = embedding_dim

        # Text stem tower
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name, trust_remote_code=True
        )

        # Document vision tower
        self.doc_vision_encoder = AutoModel.from_pretrained(
            vision_model_name, trust_remote_code=True
        )
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.doc_vision_encoder.parameters():
            param.requires_grad = False

        self.query_proj = ProjectionHead()
        self.doc_text_proj = ProjectionHead()
        self.doc_vision_proj = ProjectionHead()
        self.to(self.device)

    def mean_pool_normalize(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        text_embedding = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_embedding = F.layer_norm(
            text_embedding, normalized_shape=(text_embedding.shape[1],)
        )
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        return text_embedding

    def forward(self, batch):
        # Encode the query
        queries = {
            "input_ids": batch["anchor_input_ids"],
            "attention_mask": batch["anchor_attention_mask"],
        }
        doc_texts = {
            "input_ids": batch["doc_text_input_ids"],
            "attention_mask": batch["doc_text_attention_mask"],
        }
        doc_images = {"pixel_values": batch["doc_vision_pixel_values"]}

        with torch.no_grad():
            query_outputs = self.text_encoder(**queries)
            doc_text_outputs = self.text_encoder(**doc_texts)
        query_embedding = self.mean_pool_normalize(
            query_outputs, queries["attention_mask"]
        )
        doc_text_embedding = self.mean_pool_normalize(
            doc_text_outputs, doc_texts["attention_mask"]
        )

        # Encode the document vision
        doc_vision_outputs = self.doc_vision_encoder(**doc_images).last_hidden_state
        doc_vision_embedding = F.normalize(doc_vision_outputs[:, 0], p=2, dim=1)

        pro_query_embedding = self.query_proj(query_embedding)
        pro_doc_text_embedding = self.doc_text_proj(doc_text_embedding)
        pro_doc_vision_embedding = self.doc_vision_proj(doc_vision_embedding)
        pro_doc_text_vision_embedding = (
            pro_doc_text_embedding * pro_doc_vision_embedding
        )

        pro_doc_embedding = torch.cat(
            [
                pro_doc_text_embedding,
                pro_doc_vision_embedding,
                pro_doc_text_vision_embedding,
            ],
            dim=1,
        )
        pro_doc_embedding = nn.Dropout(p=0.3)(pro_doc_embedding)
        pro_doc_embedding = nn.Linear(self.embedding_dim * 3, self.embedding_dim)(
            pro_doc_embedding
        )

        return (
            pro_query_embedding,
            pro_doc_embedding,
            query_embedding,
            doc_text_embedding,
            doc_vision_embedding,
        )


def main():
    text_model_name = "nomic-ai/nomic-embed-text-v1.5"
    vision_model_name = "nomic-ai/nomic-embed-vision-v1.5"
    embedding_dim = 256
    p = Preprocessor(
        text_model_name,
        vision_model_name,
    )

    model = ThreeTowerRetrievalModel(text_model_name, vision_model_name, embedding_dim)

    queries = [
        "search_query: What are cute animals to cuddle with?",
        "search_query: Why do people like raccoons?",
    ]
    doc_texts = [
        "search_document: Kittens all the way to the moon",
        "search_document: Raccoons are cute and cuddly",
    ]

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "https://media.istockphoto.com/id/1272108713/photo/little-raccoon-face-looking-through-wooden-deck-rails.jpg?s=612x612&w=0&k=20&c=UhlpTfx66zFDmqloqXAqy7S9Uq7bE_Sy9CzjjZmnouM=",
    ]
    doc_images = [Image.open(requests.get(u, stream=True).raw) for u in urls]
    anchor = p.text_processor(
        queries,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    doc_texts = p.text_processor(
        doc_texts,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    doc_images = p.image_processor(doc_images, return_tensors="pt")
    inputs = {
        "anchor_input_ids": anchor["input_ids"],
        "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
        "doc_text_input_ids": doc_texts["input_ids"],
        "doc_text_attention_mask": doc_texts["attention_mask"].squeeze(0),
        "doc_vision_pixel_values": doc_images["pixel_values"].squeeze(0),
    }
    query_embedding, doc_embedding, _, _, _ = model(inputs)

    click.secho(f"Query Embedding Shape: {query_embedding.shape}", fg="yellow")
    click.secho(f"Document Embedding Shape: {doc_embedding.shape}", fg="yellow")

    cosine_similarities = F.cosine_similarity(query_embedding, doc_embedding, dim=1)
    click.secho(f"Similarity: Query vs Doc Text: {cosine_similarities}", fg="green")


if __name__ == "__main__":
    main()
