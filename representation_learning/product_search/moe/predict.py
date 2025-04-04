import click
import torch

from dataloader import TripletDataLoader
from transformers import AutoTokenizer
from transformers import AutoModel, AutoConfig

from train import evaluate_model

torch.cuda.empty_cache()


@click.command()
@click.option("--batch_size", default=512, help="Batch size for training")
def main(batch_size):
    # Set device
    config = AutoConfig.from_pretrained("lv12/bert_base_uncased_embedding_moe", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "lv12/bert_base_uncased_embedding_moe",
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.secho(f"Using device: {device}", fg="blue")

    # Load dataset
    click.secho("Loading the dataset.", fg="yellow")
    # Initialize tokenizer (using BERT tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small", trust_remote_code=True)
    dl = TripletDataLoader(tokenizer, batch_size=batch_size)
    test_loader = dl.load(split="test")

    model.to(device)

    # Evaluate the model
    click.secho("Evaluating model.", fg="yellow")
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()
