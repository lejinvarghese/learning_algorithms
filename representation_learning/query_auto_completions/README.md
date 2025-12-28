# Query Auto-Completion with Search Intention Network

Simplified implementation inspired by "Search Intention Network for Personalized Query Auto-Completion in E-Commerce" (2024).

## Architecture Overview

```
Prefix Input → Embedding → CNN + Transformer → Prefix Intention
                                                      ↓
Candidate Input → Embedding → Transformer → Candidate Intention
                                                      ↓
                                        [Concat] → MLP → Match Score
```

**Simplifications from original paper:**
- Removed historical behavior sequences and intention transfer module
- Focus on prefix-candidate matching using IE (Intention Equivocality) module
- Uses ByT5 tokenizer with optional pretrained embeddings

## Key Components

### 1. Asymmetric Encoders

**PrefixEncoder**: CNN + Transformer
- Multi-scale 1D CNN (kernel sizes: 3, 4, 5) extracts local patterns
- Transformer processes CNN output with pre-norm architecture
- Captures character-level patterns in typed prefixes

**CandidateEncoder**: Transformer only
- Pure transformer with max-pooling
- No CNN needed for complete queries
- Processes full candidate sequences

### 2. ByT5 Tokenizer
- Byte-level tokenization (259 vocab size)
- Character-aware, handles typos and rare words
- Optional pretrained embeddings from ByT5-small

### 3. Match Predictor
- 3-layer MLP with GELU activation
- LayerNorm + Dropout for stability
- Outputs match probability (0-1)

## Usage

```python
from model import QueryCompletionModel

model = QueryCompletionModel(
    vocab_size=259,  # ByT5 vocab size
    embed_dim=256,
    num_filters=64,
    num_heads=4,
    num_transformer_layers=2,
    use_pretrained_embeddings=True,  # Use ByT5 pretrained embeddings
    pretrained_model_name="google/byt5-small"
)

# Forward pass
score = model(prefix_ids, candidate_ids)  # Returns match probability
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