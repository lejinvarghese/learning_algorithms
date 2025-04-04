import click
import torch
from transformers import AutoModel

MODEL_NAME = "BAAI/BGE-VL-base"  # or "BAAI/BGE-VL-large"

model = AutoModel.from_pretrained(
    MODEL_NAME, trust_remote_code=True
)  # You must set trust_remote_code=True
model.set_processor(MODEL_NAME)
model.eval()

query_texts = [
    "something that looks like this one",
    "grey sweatpants",
    "a girl with a black cap",
]
query_images = ["./assets/blue_hoodie.png", None, None]

with torch.no_grad():
    queries = []
    for query_image, query_text in zip(query_images, query_texts):
        if query_image is None:
            query = model.encode(text=query_text)
        else:
            query = model.encode(images=query_image, text=query_text)
        queries.append(query)
    click.secho(f"Embedding Dimension: {len(queries[0][0])}", fg="blue")
    queries = torch.cat(queries)

    candidates = model.encode(
        images=[
            "./assets/blue_shorts.png",
            "./assets/green_hoodie.png",
            "./assets/green_tee.png",
            "./assets/grey_pants.png",
        ],
        text=[
            "a pair of shorts",
            "a hoodie",
            "a tshirt",
            "light gray sweatpants",
        ],
    )

    scores = queries @ candidates.T

    click.secho(scores, fg="yellow")
