import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests


class ThreeTowerRetrievalModel(nn.Module):
    def __init__(
        self, text_model_name: str, vision_model_name: str, embedding_dim: int
    ):
        super().__init__()

        # Text stem tower
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name, trust_remote_code=True
        )
        self.text_encoder.eval()

        # Document vision tower
        self.doc_vision_processor = AutoImageProcessor.from_pretrained(
            vision_model_name
        )
        self.doc_vision_encoder = AutoModel.from_pretrained(
            vision_model_name, trust_remote_code=True
        )
        self.doc_vision_encoder.eval()

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.doc_vision_encoder.parameters():
            param.requires_grad = False

        # Linear layers to project into a common embedding space
        self.query_proj = nn.Linear(self.text_encoder.config.hidden_size, embedding_dim)
        self.doc_text_proj = nn.Linear(
            self.text_encoder.config.hidden_size, embedding_dim
        )
        self.doc_vision_proj = nn.Linear(
            self.doc_vision_encoder.config.hidden_size, embedding_dim
        )

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

    def forward(self, queries, doc_texts, doc_images):
        # Encode the query

        encoded_queries = self.text_tokenizer(
            queries, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_doc_texts = self.text_tokenizer(
            doc_texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            query_outputs = self.text_encoder(**encoded_queries)
            doc_text_outputs = self.text_encoder(**encoded_doc_texts)
        query_embedding = self.mean_pool_normalize(
            query_outputs, encoded_queries["attention_mask"]
        )
        doc_text_embedding = self.mean_pool_normalize(
            doc_text_outputs, encoded_doc_texts["attention_mask"]
        )

        # Encode the document vision
        doc_vision_inputs = self.doc_vision_processor(doc_images, return_tensors="pt")
        print(doc_vision_inputs.keys())
        doc_vision_outputs = self.doc_vision_encoder(
            doc_vision_inputs["pixel_values"]
        ).last_hidden_state
        doc_vision_embedding = F.normalize(doc_vision_outputs[:, 0], p=2, dim=1)

        # # Combine document embeddings (e.g., sum or concatenate)
        project_query_embedding = self.query_proj(query_embedding)
        project_doc_text_embedding = self.doc_text_proj(doc_text_embedding)
        project_doc_vision_embedding = self.doc_vision_proj(doc_vision_embedding)
        # doc_embedding = F.concat(doc_text_embedding + doc_vision_embedding)

        return (
            project_query_embedding,
            project_doc_text_embedding,
            project_doc_vision_embedding,
        )


if __name__ == "__main__":
    text_model_name = "nomic-ai/nomic-embed-text-v1.5"
    vision_model_name = "nomic-ai/nomic-embed-vision-v1.5"
    embedding_dim = 256

    model = ThreeTowerRetrievalModel(text_model_name, vision_model_name, embedding_dim)

    queries = [
        "search_query: What are cute animals to cuddle with?",
        "search_query: What do cats look like?",
    ]
    documents = [
        "search_document: Cats and raccoons",
        "search_document: Sweeter than raccoons",
    ]

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        "http://images.cocodataset.org/val2017/000000039769.jpg",
    ]
    images = [Image.open(requests.get(u, stream=True).raw) for u in urls]

    query_embedding, doc_text_embedding, doc_vision_embedding = model(
        queries, documents, images
    )

    click.secho(f"Query Embedding Shape: {query_embedding.shape}", fg="yellow")
    click.secho(
        f"Document Text Embedding Shape: {doc_text_embedding.shape}", fg="yellow"
    )
    click.secho(
        f"Document Vision Embedding Shape: {doc_vision_embedding.shape}", fg="yellow"
    )
