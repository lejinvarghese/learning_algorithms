import click
from tqdm import tqdm

import torch
from transformers import get_scheduler
from data_loader import Preprocessor
from model import ThreeTowerRetrievalModel
from losses import MultipleNegativesSymmetricRankingLoss
from metrics import compute_metrics


@click.command()
@click.option("--batch_size", default=16, help="Batch size")
@click.option("--sample_size", default=None, help="Sample size")
@click.option("--embedding_dim", default=768, help="Embedding dimension")
@click.option("--output_dir", default="outputs", help="Output directory")
@click.option("--accumulation_steps", default=4, help="Accumulation steps")
@click.option("--log_steps", default=10, help="Log steps")
def main(
    batch_size, sample_size, embedding_dim, output_dir, accumulation_steps, log_steps
):
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
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
    total_pos_sim = 0
    total_text_alignment = 0
    total_image_alignment = 0
    for epoch in range(n_epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader),
            desc=f"Training: Epoch: {epoch}, Step ",
            colour="green",
        ):
            (
                anchor,
                positive,
                reference_anchor,
                reference_positive_text,
                reference_positive_vision,
            ) = model(batch)
            loss = criterion(
                anchor,
                positive,
                reference_anchor,
                reference_positive_text,
                reference_positive_vision,
            )
            loss.backward()

            pos_sim, text_alignment, image_alignment = compute_metrics(
                anchor, positive, reference_positive_text, reference_positive_vision
            )
            total_loss += loss.item()
            total_pos_sim += pos_sim
            total_text_alignment += text_alignment
            total_image_alignment += image_alignment

            if step % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % log_steps == 0:
                click.secho(f"Loss: {total_loss / (step + 1):.4f}", fg="red")
                click.secho(
                    f"Positive Similarity: {total_pos_sim / (step + 1):.4f}, Text Alignment: {total_text_alignment / (step + 1):.4f}, Vision Alignment: {total_image_alignment / (step + 1):.4f}",
                    fg="green",
                )
        torch.save(model.state_dict(), f"{output_dir}")


if __name__ == "__main__":
    main()
