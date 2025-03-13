from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import random

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
        filtered_data = [(entry["premise"], entry["hypothesis"], entry["label"]) 
                         for entry in dataset 
                         if entry["label"] != -1 and entry["premise"] and entry["hypothesis"]]
        
        # Create triplets: (anchor, positive, negative)
        # For each sentence pair with "entailment" label, find a negative example
        entailment_pairs = [(p, h) for p, h, label in filtered_data if label == 0]  # 0 is entailment in SNLI
        contradiction_pairs = [(p, h) for p, h, label in filtered_data if label == 2]  # 2 is contradiction in SNLI
        
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
        anchor_encoding = self.tokenizer(anchor, max_length=self.max_length, 
                                         padding='max_length', truncation=True, return_tensors='pt')
        positive_encoding = self.tokenizer(positive, max_length=self.max_length, 
                                           padding='max_length', truncation=True, return_tensors='pt')
        negative_encoding = self.tokenizer(negative, max_length=self.max_length, 
                                           padding='max_length', truncation=True, return_tensors='pt')
        
        # Return tensors of input_ids, attention_mask
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0)
        }

# Expert class using pre-trained BERT
class BERTExpert(nn.Module):
    def __init__(self, model_name, output_dim):
        super(BERTExpert, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # Freeze BERT parameters for efficiency (optional)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Projection layer to get the final embedding
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT output (last hidden state)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token embedding as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Project to desired dimension
        embedding = self.projection(cls_embedding)
        return embedding

# Gating Network
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

# Mixture of Experts for sentence embeddings using BERT
class BERTEmbeddingMoE(nn.Module):
    def __init__(self, output_dim=128, num_experts=2):
        super(BERTEmbeddingMoE, self).__init__()
        # Two different varieties of BERT for our experts
        self.expert1 = BERTExpert("bert-base-uncased", output_dim)
        self.expert2 = BERTExpert("bert-base-uncased", output_dim)
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
        mixed_output = gating_output[:, 0].unsqueeze(1) * expert1_output + \
                       gating_output[:, 1].unsqueeze(1) * expert2_output
        
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
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move all inputs to device
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            
            # Get embeddings for all three sentences
            anchor_embedding = model(anchor_input_ids, anchor_attention_mask)
            positive_embedding = model(positive_input_ids, positive_attention_mask)
            negative_embedding = model(negative_input_ids, negative_attention_mask)
            
            # Compute loss
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move all inputs to device
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            
            # Get embeddings
            anchor_embedding = model(anchor_input_ids, anchor_attention_mask)
            positive_embedding = model(positive_input_ids, positive_attention_mask)
            negative_embedding = model(negative_input_ids, negative_attention_mask)
            
            # Calculate distances
            positive_distance = (anchor_embedding - positive_embedding).pow(2).sum(1)
            negative_distance = (anchor_embedding - negative_embedding).pow(2).sum(1)
            
            # Check if positive is closer than negative
            correct += (positive_distance < negative_distance).sum().item()
            total += anchor_embedding.size(0)
    
    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading SNLI dataset...")
    dataset = load_snli_dataset()
    
    # Initialize tokenizer (using BERT tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create train and validation datasets
    print("Preparing triplet datasets...")
    train_dataset = SentenceTripletDataset(dataset["train"], tokenizer)
    val_dataset = SentenceTripletDataset(dataset["validation"], tokenizer)
    
    # Create data loaders
    batch_size = 16  # Smaller batch size due to BERT memory requirements
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    output_dim = 128
    model = BERTEmbeddingMoE(output_dim=output_dim)
    model.to(device)
    
    # Initialize loss and optimizer
    criterion = TripletLoss(margin=0.5)
    
    # Only optimize the projection layers and gating network
    optimizer_params = list(model.expert1.projection.parameters()) + \
                      list(model.expert2.projection.parameters()) + \
                      list(model.gating.parameters())
    
    optimizer = torch.optim.Adam(optimizer_params, lr=0.0001)
    
    # Train the model
    print("Starting training...")
    model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=3)
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, val_loader, device)
    
    # Save the model
    torch.save(model.state_dict(), f"models/bert_embedding_moe.pt")
    print("Model saved as bert_embedding_moe.pt")