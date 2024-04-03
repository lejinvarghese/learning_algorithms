import click
from gliner import GLiNER
import torch
from warnings import filterwarnings

filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GLiNER.from_pretrained("urchade/gliner_base", map_location=device)

tuned_model = GLiNER.from_pretrained("logs/finetuned_999", map_location=device)

queries = [
    "apple iphone 15 black with 256 GB storage IRU5121",
    "apple cider vinegar",
    "apple macbook pro 2021",
    "nikon d3500 camera",
    "adidas samba shoes",
    "cate blanchett armani perfume",
    "purple mattress",
    "10 inch amzon fire",
    "amazon fire tv",
    "nike air max 2021",
    "12 pack bang energy drink",
]

labels = [
    # "product",
    # "product taxonomy",
    # "product category",
    # "sku",
    "brand",
    # "organization",
    # "person",
]


@click.command()
@click.option(
    "--threshold", type=float, default=0.5, help="Threshold for entity recognition."
)
def main(threshold):
    for q in queries:
        click.secho(f"\nQuery: {q}", fg="red")
        entities = model.predict_entities(q, labels, threshold=threshold)
        tuned_entities = tuned_model.predict_entities(q, labels, threshold=threshold)
        for entity in entities:
            click.secho(f'Raw model:: {entity["text"]} => {entity["label"]}', fg="blue")
        for entity in tuned_entities:
            click.secho(
                f'Tuned model:: {entity["text"]} => {entity["label"]}', fg="yellow"
            )


if __name__ == "__main__":
    main()
