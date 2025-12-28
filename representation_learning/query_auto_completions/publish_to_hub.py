"""
Publish Query Auto-Completion model to HuggingFace Hub

This script exports the trained model in a HuggingFace-compatible format,
allowing it to be loaded via AutoModel and AutoConfig.
"""

import os
import torch
import click
from pathlib import Path
from typing import Union, Tuple

from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

# Import your existing model definition
from model import QueryCompletionModel as BaseQueryCompletionModel


class QueryCompletionConfig(PretrainedConfig):
    """Configuration for Query Auto-Completion model."""

    model_type = "query-completion"

    def __init__(
        self,
        vocab_size: int = 384,
        embed_dim: int = 256,
        num_filters: int = 64,
        filter_sizes: list = None,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        use_pretrained_embeddings: bool = True,
        pretrained_model_name: str = "google/byt5-small",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes or [3, 4, 5]
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_model_name = pretrained_model_name
        self.dropout = dropout


class QueryCompletionModelForHub(PreTrainedModel):
    """
    HuggingFace wrapper around the Query Auto-Completion model.

    Wraps the existing model.py implementation for Hub compatibility.
    """

    config_class = QueryCompletionConfig
    base_model_prefix = "query_completion"
    supports_gradient_checkpointing = False

    def __init__(self, config: QueryCompletionConfig):
        super().__init__(config)

        # Use the existing model implementation
        self.model = BaseQueryCompletionModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_filters=config.num_filters,
            num_heads=config.num_heads,
            num_transformer_layers=config.num_transformer_layers,
            use_pretrained_embeddings=config.use_pretrained_embeddings,
            pretrained_model_name=config.pretrained_model_name,
        )

        self.post_init()

    def forward(
        self,
        prefix_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        return_dict: bool = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass to compute match scores.

        Args:
            prefix_ids: Token IDs for the query prefix (batch_size, seq_len)
            candidate_ids: Token IDs for the candidate completion (batch_size, seq_len)
            return_dict: Unused, for API compatibility

        Returns:
            Match scores between 0 and 1 (batch_size, 1)
        """
        return self.model(prefix_ids, candidate_ids)


def load_checkpoint_weights(checkpoint_path: str) -> Tuple[dict, dict]:
    """Load weights from PyTorch Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    # Remove 'model.' prefix from keys (Lightning wraps in 'model.')
    # Our wrapper also uses 'model.', so we keep one level
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key] = value
        else:
            new_state_dict[f"model.{key}"] = value

    return new_state_dict, checkpoint.get("hyper_parameters", {})


def create_model_card(
    repo_id: str,
    config: QueryCompletionConfig,
    checkpoint_path: str,
    dataset_name: str = "rexoscare/autocomplete-search-dataset",
) -> str:
    """Generate model card content."""
    return f'''---
license: apache-2.0
library_name: transformers
tags:
  - query-auto-completion
  - search
  - pytorch
  - custom-model
datasets:
  - {dataset_name}
language:
  - en
pipeline_tag: text-classification
---

# Query Auto-Completion Model

A CNN+Transformer model for ranking query auto-completion suggestions.

## Model Description

This model scores how well a candidate completion matches a given query prefix. It uses:
- **Prefix Encoder**: Multi-scale CNN + Transformer for extracting search intention from partial queries
- **Candidate Encoder**: Transformer for encoding candidate completions
- **Match Predictor**: MLP that scores the compatibility between prefix and candidate

The model uses pretrained ByT5 byte-level embeddings for robust character-level understanding.

## Model Architecture

- Embedding Dimension: {config.embed_dim}
- CNN Filters: {config.num_filters} (filter sizes: {config.filter_sizes})
- Transformer Heads: {config.num_heads}
- Transformer Layers: {config.num_transformer_layers}
- Base Embeddings: {config.pretrained_model_name}

## Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from transformers import AutoTokenizer, AutoConfig, AutoModel

# Load model (trust_remote_code required for custom architecture)
config = AutoConfig.from_pretrained("{repo_id}", trust_remote_code=True)
model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
```

### Scoring Candidates

```python
import torch

def score_completion(model, tokenizer, prefix: str, candidate: str, max_length: int = 20):
    """Score how well a candidate matches a prefix."""
    model.eval()

    prefix_encoding = tokenizer(
        prefix,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    candidate_encoding = tokenizer(
        candidate,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        score = model(
            prefix_ids=prefix_encoding["input_ids"],
            candidate_ids=candidate_encoding["input_ids"]
        )

    return score.squeeze().item()

# Example usage
prefix = "how to"
candidates = ["how to cook pasta", "how to learn python", "weather today"]

scores = []
for candidate in candidates:
    score = score_completion(model, tokenizer, prefix, candidate)
    scores.append((candidate, score))

# Sort by score (higher is better match)
scores.sort(key=lambda x: x[1], reverse=True)
for candidate, score in scores:
    print(f"{{score:.4f}} - {{candidate}}")
```

### Batch Scoring

```python
def batch_score(model, tokenizer, prefix: str, candidates: list, max_length: int = 20):
    """Score multiple candidates efficiently."""
    model.eval()

    prefix_encoding = tokenizer(
        prefix,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    prefix_ids = prefix_encoding["input_ids"]

    candidate_encodings = tokenizer(
        candidates,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    scores = []
    with torch.no_grad():
        for i in range(len(candidates)):
            score = model(
                prefix_ids=prefix_ids,
                candidate_ids=candidate_encodings["input_ids"][i:i+1]
            )
            scores.append(score.squeeze().item())

    return list(zip(candidates, scores))

# Example
results = batch_score(model, tokenizer, "best resta", [
    "best restaurants near me",
    "best restaurant in new york",
    "best resume templates",
    "weather forecast"
])
for candidate, score in sorted(results, key=lambda x: -x[1]):
    print(f"{{score:.4f}} - {{candidate}}")
```

## Training Details

- **Dataset**: [{dataset_name}](https://huggingface.co/datasets/{dataset_name})
- **Checkpoint**: `{Path(checkpoint_path).name}`
- **Training Framework**: PyTorch Lightning
- **Validation Loss**: 0.0459

## Evaluation

The model outputs scores between 0 and 1:
- **> 0.7**: Strong match (candidate is a likely completion)
- **0.4 - 0.7**: Moderate match
- **< 0.4**: Weak match (candidate unlikely to be what user wants)

## Limitations

- Optimized for English queries
- Best performance on short prefixes (< 20 characters)
- Trained on search autocomplete data; may not generalize to other domains

## Citation

If you use this model, please cite:

```bibtex
@misc{{query-completion-model,
  title={{Query Auto-Completion Model}},
  year={{2024}},
  publisher={{HuggingFace}},
  url={{https://huggingface.co/{repo_id}}}
}}
```
'''


@click.command()
@click.option(
    "--checkpoint",
    default="./lightning_logs/sin_qac/version_5/checkpoints/final.ckpt",
    help="Path to model checkpoint",
)
@click.option(
    "--repo_id",
    default="lv12/sin-qac-model",
    help="HuggingFace repo ID (e.g., 'username/model-name')",
)
@click.option(
    "--output_dir",
    default="./hf_model_export",
    help="Local directory to save model files",
)
@click.option(
    "--push_to_hub",
    is_flag=True,
    default=False,
    help="Push to HuggingFace Hub",
)
@click.option(
    "--private",
    is_flag=True,
    default=False,
    help="Make the repo private",
)
@click.option(
    "--dataset_name",
    default="rexoscare/autocomplete-search-dataset",
    help="Dataset used for training (for model card)",
)
def main(checkpoint, repo_id, output_dir, push_to_hub, private, dataset_name):
    """Publish Query Auto-Completion model to HuggingFace Hub."""

    click.secho("\n" + "=" * 60, fg="bright_cyan", bold=True)
    click.secho("  Query Auto-Completion Model Publisher", fg="bright_cyan", bold=True)
    click.secho("=" * 60 + "\n", fg="bright_cyan", bold=True)

    # Register custom model with Auto classes
    AutoConfig.register("query-completion", QueryCompletionConfig)
    AutoModel.register(QueryCompletionConfig, QueryCompletionModelForHub)

    # Load checkpoint
    click.secho(f"Loading checkpoint: {checkpoint}", fg="cyan")
    if not os.path.exists(checkpoint):
        click.secho(f"Checkpoint not found: {checkpoint}", fg="red")
        return

    state_dict, hparams = load_checkpoint_weights(checkpoint)
    click.secho(f"  Loaded {len(state_dict)} weight tensors", fg="green")
    click.secho(f"  Hyperparameters: {hparams}", fg="yellow")

    # Create config from hyperparameters
    config = QueryCompletionConfig(
        vocab_size=hparams.get("vocab_size", 384),
        embed_dim=hparams.get("embed_dim", 256),
        num_filters=64,
        num_heads=4,
        num_transformer_layers=2,
        use_pretrained_embeddings=hparams.get("use_pretrained_embeddings", True),
        pretrained_model_name=hparams.get("pretrained_model_name", "google/byt5-small"),
    )

    # Initialize model
    click.secho("\nInitializing model...", fg="cyan")
    model = QueryCompletionModelForHub(config)

    # Load weights
    click.secho("Loading trained weights...", fg="cyan")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        click.secho(f"  Missing keys: {missing}", fg="yellow")
    if unexpected:
        click.secho(f"  Unexpected keys: {unexpected}", fg="yellow")
    click.secho("  Weights loaded successfully!", fg="green")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model and config
    click.secho(f"\nSaving to {output_path}...", fg="cyan")
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    # Copy model.py for trust_remote_code support
    import shutil

    model_py_src = Path(__file__).parent / "model.py"
    model_py_dst = output_path / "model.py"
    shutil.copy(model_py_src, model_py_dst)
    click.secho(f"  Copied model.py for remote code support", fg="green")

    # Create modeling file that HuggingFace expects
    modeling_content = '''"""
HuggingFace-compatible model definition for Query Auto-Completion.
"""

import torch
from typing import Union, Tuple
from transformers import PretrainedConfig, PreTrainedModel

from .model import QueryCompletionModel as BaseQueryCompletionModel


class QueryCompletionConfig(PretrainedConfig):
    """Configuration for Query Auto-Completion model."""

    model_type = "query-completion"

    def __init__(
        self,
        vocab_size: int = 384,
        embed_dim: int = 256,
        num_filters: int = 64,
        filter_sizes: list = None,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        use_pretrained_embeddings: bool = True,
        pretrained_model_name: str = "google/byt5-small",
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes or [3, 4, 5]
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.pretrained_model_name = pretrained_model_name
        self.dropout = dropout


class QueryCompletionModelForHub(PreTrainedModel):
    """HuggingFace wrapper around the Query Auto-Completion model."""

    config_class = QueryCompletionConfig
    base_model_prefix = "query_completion"
    supports_gradient_checkpointing = False

    def __init__(self, config: QueryCompletionConfig):
        super().__init__(config)

        self.model = BaseQueryCompletionModel(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_filters=config.num_filters,
            num_heads=config.num_heads,
            num_transformer_layers=config.num_transformer_layers,
            use_pretrained_embeddings=config.use_pretrained_embeddings,
            pretrained_model_name=config.pretrained_model_name,
        )

        self.post_init()

    def forward(
        self,
        prefix_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        return_dict: bool = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.model(prefix_ids, candidate_ids)
'''

    modeling_path = output_path / "modeling_query_completion.py"
    modeling_path.write_text(modeling_content)
    click.secho(f"  Created modeling_query_completion.py", fg="green")

    # Create and save model card
    model_card = create_model_card(repo_id, config, checkpoint, dataset_name)
    readme_path = output_path / "README.md"
    readme_path.write_text(model_card)
    click.secho(f"  Created model card: {readme_path}", fg="green")

    # Update config to reference custom code
    import json

    config_path = output_path / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    config_dict["auto_map"] = {
        "AutoConfig": "modeling_query_completion.QueryCompletionConfig",
        "AutoModel": "modeling_query_completion.QueryCompletionModelForHub",
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    click.secho(f"  Updated config.json with auto_map", fg="green")

    # List saved files
    click.secho("\nSaved files:", fg="bright_green")
    for f in sorted(output_path.iterdir()):
        size = f.stat().st_size / 1024
        click.secho(f"  - {f.name} ({size:.1f} KB)", fg="white")

    # Push to hub
    if push_to_hub:
        click.secho(f"\nPushing to HuggingFace Hub: {repo_id}...", fg="cyan")
        try:
            from huggingface_hub import HfApi, create_repo

            api = HfApi()

            # Create repo if it doesn't exist
            try:
                create_repo(repo_id, private=private, exist_ok=True)
            except Exception:
                pass

            # Upload all files
            api.upload_folder(
                folder_path=str(output_path),
                repo_id=repo_id,
                repo_type="model",
            )

            click.secho(
                f"\nModel published to: https://huggingface.co/{repo_id}",
                fg="bright_green",
                bold=True,
            )
        except Exception as e:
            click.secho(f"\nFailed to push to hub: {e}", fg="red")
            click.secho(
                "Make sure you're logged in: huggingface-cli login", fg="yellow"
            )
    else:
        click.secho(
            f"\nModel saved locally to: {output_path}", fg="bright_green", bold=True
        )
        click.secho("To push to hub, run with --push_to_hub flag", fg="yellow")

    click.secho("\nDone!", fg="bright_green", bold=True)


if __name__ == "__main__":
    main()
