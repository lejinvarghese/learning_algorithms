import click
from gliner import GLiNER
from warnings import filterwarnings

filterwarnings("ignore")

model = GLiNER.from_pretrained("urchade/gliner_largev2")

queries = [
    "apple iphone 15 black with 256 GB storage IRU5121",
    "apple cider vinegar",
    "apple macbook pro 2021",
    "nikon d3500 camera",
    "adidas samba shoes",
    "cate blanchett armani perfume",
    "purple mattress"
]

labels = [
    "product",
    "product taxonomy",
    "product category",
    "sku",
    "brand",
    "organization",
    "person",
]

@click.command()
@click.option("--threshold", type=float, default=0.5, help="Threshold for entity recognition.")
def main(threshold):   
    for q in queries:
        click.secho(f"\nQuery: {q}", fg="red")
        entities = model.predict_entities(q, labels, threshold=threshold)
        for entity in entities:
            click.secho(f'{entity["text"]} => {entity["label"]}', fg="blue")

if __name__ == "__main__":
    main()
