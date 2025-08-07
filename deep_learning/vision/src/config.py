"""
Simple configuration for FLUX/SD LoRA training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class Config:
    """Main configuration for LoRA training."""
    
    # Project
    project_name: str = "personal-taste-lora"
    output_dir: str = "outputs"
    data_dir: str = "data"
    
    # Model - simplified to FLUX/SD only
    model_name: str = "black-forest-labs/FLUX.1-dev"  # or "stabilityai/stable-diffusion-3.5-large"
    
    # LoRA settings (2025 optimized)
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    
    # Training 
    learning_rate: float = 1e-5
    batch_size: int = 1
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    
    # DeviantArt
    deviantart_url: str = ""
    deviant_client_id: str = ""
    deviant_client_secret: str = ""
    max_images: int = 200
    
    # Image processing
    resolution: int = 1024
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "lora-training"
    
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