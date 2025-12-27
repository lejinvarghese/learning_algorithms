# Search Intention Network (SIN)

Implementation of the Search Intention Network from "Search Intention Network for Personalized Query Auto-Completion in E-Commerce" (2024).

## Architecture Overview

```
Prefix Input → IE Module → Current Intention
                              ↓
Behavior Sequences → Multi-View Encoders → Historical Intentions
                              ↓
                    Candidate-to-History Attention (time decay)
                              ↓
                    IT Module → Final Intention → CTR Prediction
```

## Key Components

### 1. Intention Equivocality (IE) Module
**Problem**: Short prefixes are ambiguous during typing  
**Solution**: CNN extracts local patterns → Transformer distills intention

- `CNNLocalEncoder`: 3 parallel 1D convolutions (filters: 3,4,5) + max pooling
- `IntentionEquivocalityModule`: CNN + Transformer encoder

### 2. Multi-View Sequence Encoder
**Purpose**: Encode different behavior types (searches, clicks, purchases)

- Transformer encoder per behavior type
- Positional encoding for temporal awareness

### 3. Candidate-to-History Attention
**Purpose**: Weight historical items by relevance and recency

- Scaled dot-product attention
- Time-decaying weights (recent = higher weight)

### 4. Intention Transfer (IT) Module
**Problem**: Current intent may differ from historical preferences  
**Solution**: Measure transfer via vector distance, balance intentions

- Learns transfer score from [current, historical] concatenation
- Weighted combination: low score → current dominates, high → historical dominates

### 5. Search Intention Network (Main)
Combines all components for CTR prediction

## Usage

```python
from model import SearchIntentionNetwork

model = SearchIntentionNetwork(
    vocab_size=10000,
    embed_dim=128,
    num_behaviors=3  # searches, clicks, purchases
)

ctr_score, components = model(
    prefix_ids=prefix_ids,
    behavior_sequences=[search_seq, click_seq, purchase_seq],
    behavior_masks=[mask1, mask2, mask3],
    time_decays=[decay1, decay2, decay3]
)
```

## Training

See [QUICKSTART.md](QUICKSTART.md) for detailed commands and examples.

### Quick test (small dataset, 1 epoch)
```bash
python train.py --max_train_samples 1000 --max_val_samples 100 --epochs 1 --batch_size 16
```

### Fast test (medium dataset, 3 epochs)
```bash
python train.py --max_train_samples 5000 --max_val_samples 500 --epochs 3 --batch_size 32
```

### Full training (default settings)
```bash
python train.py --epochs 10 --batch_size 32
```

### Custom configuration
```bash
python train.py \
  --epochs 5 \
  --batch_size 64 \
  --lr 0.001 \
  --embed_dim 256 \
  --vocab_size 5000 \
  --max_train_samples 10000 \
  --max_val_samples 1000
```

## Outputs

- `ctr_score`: Click-through rate for ranking
- `components`: Intermediate representations (intentions, transfer scores, attention weights)

## Datasets

```md
rexoscare/autocomplete-search-dataset ["input", "output"]
amazon/AmazonQAC ["prefixes", "final_search_term"]
```



## Command Line Options

- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--embed_dim`: Embedding dimension (default: 128)
- `--vocab_size`: Vocabulary size (default: 10000)
- `--num_behaviors`: Number of behavior types (default: 3)
- `--prefix_len`: Prefix sequence length (default: 20)
- `--seq_len`: Behavior sequence length (default: 30)
- `--dataset_name`: HuggingFace dataset name (default: rexoscare/autocomplete-search-dataset)
- `--max_train_samples`: Limit training samples for testing (default: None)
- `--max_val_samples`: Limit validation samples for testing (default: None)
- `--val_ratio`: Validation split ratio (default: 0.1)
- `--gpus`: Number of GPUs (default: 0 for CPU)