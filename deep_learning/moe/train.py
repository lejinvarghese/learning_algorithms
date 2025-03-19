from tqdm import tqdm
import click
import os
from datetime import datetime

import torch
from torch import nn
from dataloader import TripletDataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from transformers import AutoTokenizer

from model import EmbeddingMoE, EmbeddingMoEConfig

torch.cuda.empty_cache()

model_version = datetime.now().strftime("%Y%m%d%H%M%S")


# Triplet loss for learning similarity
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def get_embeddings(model, batch, device):
    anchor_input_ids = batch["anchor_input_ids"].to(device)
    anchor_attention_mask = batch["anchor_attention_mask"].to(device)
    positive_input_ids = batch["positive_input_ids"].to(device)
    positive_attention_mask = batch["positive_attention_mask"].to(device)
    negative_input_ids = batch["negative_input_ids"].to(device)
    negative_attention_mask = batch["negative_attention_mask"].to(device)

    # Get embeddings for all three sentences
    anchor_embedding = model(anchor_input_ids, anchor_attention_mask)
    positive_embedding = model(positive_input_ids, positive_attention_mask)
    negative_embedding = model(negative_input_ids, negative_attention_mask)
    return anchor_embedding, positive_embedding, negative_embedding


def train_model(
    model, train_loader, criterion, optimizer, scheduler, device, num_epochs=5
):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move all inputs to device
            anchor, positive, negative = get_embeddings(model, batch, device)

            # Compute loss
            loss = criterion(anchor, positive, negative)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        click.secho(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move all inputs to device
            anchor, positive, negative = get_embeddings(model, batch, device)

            # Calculate distances
            positive_distance = (anchor - positive).pow(2).sum(1)
            negative_distance = (anchor - negative).pow(2).sum(1)

            # Predictions and labels
            preds = (positive_distance < negative_distance).int().cpu()
            labels = torch.ones_like(
                preds
            )  # Assuming positive is always the correct class

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

            correct += (positive_distance < negative_distance).sum().item()
            total += anchor.size(0)

    # Calculate metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Print results
    click.secho(f"Evaluation Accuracy: {accuracy:.4f}", fg="green")
    click.secho(
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}",
        fg="blue",
    )

    return accuracy, precision, recall, f1


@click.command()
@click.option("--n_epochs", default=10, help="Number of epochs to train the model")
@click.option("--batch_size", default=512, help="Batch size for training")
@click.option("--n_experts", default=2, help="Number of experts in the MoE model")
@click.option("--model_name", default="thenlper/gte-small", help="Model name")
def main(n_epochs, batch_size, n_experts, model_name):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.secho(f"Using device: {device}", fg="blue")

    # Load dataset
    click.secho("Loading the dataset.", fg="yellow")
    # Initialize tokenizer (using BERT tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create train and validation datasets
    click.secho("Preparing triplet datasets.", fg="yellow")
    dl = TripletDataLoader(tokenizer, batch_size=batch_size)
    # Create data loaders
    train_loader, val_loader = (
        dl.load("train"),
        dl.load("validation"),
    )

    # Initialize model
    config = EmbeddingMoEConfig(model_name=model_name, num_experts=n_experts)
    model = EmbeddingMoE(config)
    model.to(device)

    # Initialize loss and optimizer
    criterion = TripletLoss(margin=0.5)

    # Only optimize the projection layers and gating network
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params, lr=1e-3, weight_decay=0.01, eps=1e-8
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Train the model
    click.secho("Starting training.", fg="yellow")
    model = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=n_epochs,
    )

    # Evaluate the model
    click.secho("Evaluating model.", fg="yellow")
    evaluate_model(model, val_loader, device)

    # Save the model
    file_path = f"models/{model_name.replace('-', '_')}_embedding_moe/{model_version}"
    os.makedirs(file_path, exist_ok=True)
    torch.save(model.state_dict(), f"{file_path}/pytorch_model.bin")
    click.secho(f"Model saved at {file_path}", fg="green")


if __name__ == "__main__":
    main()
