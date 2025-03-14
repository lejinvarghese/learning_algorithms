from tqdm import tqdm
import random
import click
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

from model import EmbeddingMoE, EmbeddingMoEConfig

torch.cuda.empty_cache()

model_version = datetime.now().strftime("%Y%m%d%H%M%S")


# Load the SNLI dataset which contains premise, hypothesis pairs with labels
def load_snli_dataset():
    dataset = load_dataset("snli")
    return dataset


# Custom Dataset class for triplet learning
class SentenceTripletDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = []

        # Filter out entries with 'neutral' labels and missing entries
        filtered_data = [
            (entry["premise"], entry["hypothesis"], entry["label"])
            for entry in dataset
            if entry["label"] != -1 and entry["premise"] and entry["hypothesis"]
        ]

        # Create triplets: (anchor, positive, negative)
        # For each sentence pair with "entailment" label, find a negative example
        entailment_pairs = [
            (p, h) for p, h, label in filtered_data if label == 0
        ]  # 0 is entailment in SNLI
        contradiction_pairs = [
            (p, h) for p, h, label in filtered_data if label == 2
        ]  # 2 is contradiction in SNLI

        # Create triplets (anchor, positive, negative)
        for premise, hypothesis in entailment_pairs[:10_000_000]:  # Limit for memory reasons
            # The anchor is the premise
            anchor = premise
            # The positive is the entailed hypothesis
            positive = hypothesis
            # Find a random contradiction as negative
            if contradiction_pairs:
                neg_premise, neg_hypothesis = random.choice(contradiction_pairs)
                negative = neg_hypothesis
                self.triplets.append((anchor, positive, negative))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        # Tokenize all three sentences
        anchor_encoding = self.tokenizer(
            anchor,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        positive_encoding = self.tokenizer(
            positive,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        negative_encoding = self.tokenizer(
            negative,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Return tensors of input_ids, attention_mask
        return {
            "anchor_input_ids": anchor_encoding["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": negative_encoding["input_ids"].squeeze(0),
            "negative_attention_mask": negative_encoding["attention_mask"].squeeze(0),
        }


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

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move all inputs to device
            anchor, positive, negative = get_embeddings(model, batch, device)

            # Calculate distances
            positive_distance = (anchor - positive).pow(2).sum(1)
            negative_distance = (anchor - negative).pow(2).sum(1)

            # Check if positive is closer than negative
            correct += (positive_distance < negative_distance).sum().item()
            total += anchor.size(0)

    accuracy = correct / total
    click.secho(f"Evaluation Accuracy: {accuracy:.4f}", fg="green")
    return accuracy


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
    dataset = load_snli_dataset()

    # Initialize tokenizer (using BERT tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create train and validation datasets
    click.secho("Preparing triplet datasets.", fg="yellow")
    train_dataset = SentenceTripletDataset(dataset["train"], tokenizer)
    val_dataset = SentenceTripletDataset(dataset["validation"], tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    config = EmbeddingMoEConfig()
    model = EmbeddingMoE(config)
    model.to(device)

    # Initialize loss and optimizer
    criterion = TripletLoss(margin=0.5)

    # Only optimize the projection layers and gating network
    optimizer_params = (
        list(model.expert1.projection.parameters())
        + list(model.expert2.projection.parameters())
        + list(model.gating.parameters())
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01, eps=1e-8
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
    file_path = (
        f"models/{model_name.replace('-', '_')}_embedding_moe/{model_version}"
    )
    os.makedirs(file_path, exist_ok=True)
    torch.save(model.state_dict(), f"{file_path}/pytorch_model.bin")
    click.secho(f"Model saved at {file_path}", fg="green")


if __name__ == "__main__":
    main()
