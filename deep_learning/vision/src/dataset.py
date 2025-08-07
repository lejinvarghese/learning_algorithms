"""
Simple dataset for LoRA training.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LoRADataset(Dataset):
    """Simple dataset for LoRA fine-tuning."""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        resolution: int = 1024,
        caption_ext: str = ".txt"
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.caption_ext = caption_ext
        
        # Find all images
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            self.image_paths.extend(self.data_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.data_dir.glob(f"*{ext.upper()}"))
            
        logger.info(f"Found {len(self.image_paths)} images in {data_dir}")
        
        # Setup transforms
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Apply transforms
        pixel_values = self.transforms(image)
        
        # Load caption
        caption = self._load_caption(image_path)
        
        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0),
            "caption": caption
        }
        
    def _load_caption(self, image_path: Path) -> str:
        """Load caption from corresponding text file."""
        caption_path = image_path.with_suffix(self.caption_ext)
        
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            # Fallback to filename
            caption = image_path.stem.replace('_', ' ').replace('-', ' ')
            
        return caption if caption else "an image"