import click
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests

processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained(
    "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(image, return_tensors="pt")

img_emb = vision_model(**inputs).last_hidden_state
img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

text_embedding = img_embeddings[0, :]
click.secho(f"Image Embedding: {img_embeddings.shape}", fg="green")

sim = torch.matmul(img_embeddings, text_embedding.T)


click.secho(f"Image Embedding: {sim}", fg="green")
