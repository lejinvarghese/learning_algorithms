import click
from tqdm import tqdm

import torch
from transformers import AdamW, get_scheduler
from data_loader import Preprocessor
from model import ThreeTowerRetrievalModel
from losses import MultipleNegativesSymmetricRankingLoss
from metrics import compute_metrics


@click.command()
@click.option("--batch_size", default=4, help="Batch size")
@click.option("--sample_size", default=None, help="Sample size")
@click.option("--embedding_dim", default=768, help="Embedding dimension")
@click.option("--output_dir", default="outputs", help="Output directory")
@click.option("--log_steps", default=10, help="Log steps")
def main(batch_size, sample_size, embedding_dim, output_dir, log_steps):
    text_model_name = "nomic-ai/nomic-embed-text-v1.5"
    vision_model_name = "nomic-ai/nomic-embed-vision-v1.5"
    dataloader = Preprocessor(
        text_model_name,
        vision_model_name,
        sample_size=sample_size,
        batch_size=batch_size,
    )
    train_dataloader = dataloader.run()
    model = ThreeTowerRetrievalModel(text_model_name, vision_model_name, embedding_dim)
    criterion = MultipleNegativesSymmetricRankingLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    n_epochs = 3
    n_steps = n_epochs * len(train_dataloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=n_steps,
    )
    model.train()
    total_loss = 0
    total_pos_text_sim = 0
    total_pos_vision_sim = 0
    total_modality_alignment = 0
    for epoch in tqdm(range(n_epochs), desc="Training: Epoch: ", colour="yellow"):
        for step, batch in tqdm(
            enumerate(train_dataloader), desc="Training: Step ", colour="yellow"
        ):
            anchor, positive_text, positive_vision = model(batch)
            loss = criterion(anchor, positive_text, positive_vision)
            loss.backward()

            pos_text_sim, pos_vision_sim, modality_alignment = compute_metrics(
                anchor, positive_text, positive_vision
            )
            total_loss += loss.item()
            total_pos_text_sim += pos_text_sim
            total_pos_vision_sim += pos_vision_sim
            total_modality_alignment += modality_alignment

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % log_steps == 0:
                click.secho(f"Loss: {total_loss / (step + 1)}", fg="red")
                click.secho(
                    f"Positive Text Similarity: {total_pos_text_sim / (step + 1)}, Positive Vision Similarity: {total_pos_vision_sim / (step + 1)}, Modality Alignment: {total_modality_alignment / (step + 1)}",
                    fg="green",
                )

    torch.save(model.state_dict(), f"{output_dir}")


if __name__ == "__main__":
    main()
