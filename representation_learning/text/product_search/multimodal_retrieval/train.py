import click
import torch
from data_loader import Preprocessor


@click.command()
@click.option("--batch_size", default=32, help="Batch size")
@click.option("--sample_size", default=None, help="Sample size")
def main(batch_size, sample_size):
    text_model_name = "nomic-ai/nomic-embed-text-v1.5"
    vision_model_name = "nomic-ai/nomic-embed-vision-v1.5"
    dataloader = Preprocessor(
        text_model_name,
        vision_model_name,
        sample_size=sample_size,
        batch_size=batch_size,
    )
    train_dataset = dataloader.run()
    for batch in train_dataset:
        print(batch)
        break


if __name__ == "__main__":
    main()
