"""
Configuration for FLUX LoRA training.
"""

import os
from dataclasses import dataclass, field
import yaml


@dataclass
class Config:
    """Configuration for FLUX LoRA aesthetic training."""

    # Project
    project_name: str = "flux-aesthetic-lora"
    output_dir: str = "outputs"
    data_dir: str = "data"

    # Model - FLUX only
    model_name: str = "black-forest-labs/FLUX.1-dev"

    # LoRA settings (optimized for 2025)
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 1
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"  # bf16, fp16, or no

    # DeviantArt scraping
    deviantart_url: str = ""
    deviant_client_id: str = ""
    deviant_client_secret: str = ""
    max_images: int = 200

    # Image processing
    resolution: int = 1024
    use_bucketing: bool = True  # Preserve aspect ratios
    bucket_step: int = 64

    # Captioning strategy
    caption_mode: str = "trigger"  # "trigger" (aesthetic triggers only) or "blip" (BLIP + triggers)

    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "flux-lora-training"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str):
        """Save to YAML file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def validate(self):
        """Validate configuration."""
        errors = []

        # Check required fields for scraping
        if not self.deviantart_url and not os.path.exists(os.path.join(self.data_dir, "raw")):
            errors.append("deviantart_url is required if no existing data")

        if not self.deviant_client_id or not self.deviant_client_secret:
            if not os.path.exists(os.path.join(self.data_dir, "raw")):
                errors.append("DeviantArt credentials required for scraping")

        # Check model name
        if "flux" not in self.model_name.lower():
            errors.append(f"Only FLUX models supported, got: {self.model_name}")

        # Check mixed precision
        if self.mixed_precision not in ["bf16", "fp16", "no"]:
            errors.append(f"mixed_precision must be 'bf16', 'fp16', or 'no', got: {self.mixed_precision}")

        # Check caption mode
        if self.caption_mode not in ["trigger", "blip"]:
            errors.append(f"caption_mode must be 'trigger' or 'blip', got: {self.caption_mode}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True
