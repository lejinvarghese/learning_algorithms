from tqdm import tqdm
import random
import click
from datetime import datetime
import torch.nn.functional as F


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

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
        for premise, hypothesis in entailment_pairs[:10000]:  # Limit for memory reasons
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


# Expert class using pre-trained BERT
class EmbeddingExpert(nn.Module):
    def __init__(self, model_name, output_dim, dropout_rate=0.1):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.layer_norm = nn.LayerNorm(self.base.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        for param in self.base.parameters():
            param.requires_grad = False

        # Projection layer to get the final embedding
        self.projection = nn.Linear(self.base.config.hidden_size, output_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling - take attention mask into account for averaging
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        embedding = self.projection(pooled_output)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


# Gating Network
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = torch.clamp(x, min=-10, max=10)
        x = self.softmax(x)
        return x


# Mixture of Experts for sentence embeddings using BERT
class EmbeddingMoE(nn.Module):
    def __init__(self, output_dim=128, num_experts=2):
        super().__init__()
        # Two different varieties of BERT for our experts
        self.expert1 = EmbeddingExpert("bert-base-uncased", output_dim)
        self.expert2 = EmbeddingExpert("bert-base-uncased", output_dim)
        self.gating = GatingNetwork(output_dim, 256, num_experts)
        self.output_dim = output_dim

    def forward(self, input_ids, attention_mask):
        # Get embeddings from both experts
        expert1_output = self.expert1(input_ids, attention_mask)
        expert2_output = self.expert2(input_ids, attention_mask)

        # Average the output as input to gating
        gating_input = (expert1_output + expert2_output) / 2

        # Get gating weights
        gating_output = self.gating(gating_input)

        # Combine expert outputs
        mixed_output = (
            gating_output[:, 0].unsqueeze(1) * expert1_output
            + gating_output[:, 1].unsqueeze(1) * expert2_output
        )

        # Normalize the embedding to unit length
        embedding = torch.nn.functional.normalize(mixed_output, p=2, dim=1)

        return embedding

    def encode_sentence(self, input_ids, attention_mask):
        """Helper method to get the embedding for a single sentence"""
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)


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
@click.option("--n_epochs", default=1, help="Number of epochs to train the model")
@click.option(
    "--output_dim", default=256, help="Output dimension of the sentence embeddings"
)
@click.option("--batch_size", default=512, help="Batch size for training")
@click.option("--n_experts", default=2, help="Number of experts in the MoE model")
@click.option("--model_name", default="bert-base-uncased", help="Model name")
def main(n_epochs, output_dim, batch_size, n_experts, model_name):
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
    model = EmbeddingMoE(output_dim=output_dim, num_experts=n_experts)
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
        model, train_loader, criterion, optimizer, scheduler, device, num_epochs=n_epochs
    )

    # Evaluate the model
    click.secho("Evaluating model.", fg="yellow")
    evaluate_model(model, val_loader, device)

    # Save the model
    file_path = (
        f"models/{model_name.replace('-', '_')}_embedding_moe_{model_version}.pt"
    )
    torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    main()
