"""
Dataset for FLUX LoRA training with dual tokenizer support and aspect ratio bucketing.
"""

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Any, Tuple
import logging
import math

logger = logging.getLogger(__name__)


def get_bucket_for_image(width: int, height: int, base_resolution: int = 1024, bucket_step: int = 64) -> Tuple[int, int]:
    """
    Get the best bucket size for an image to preserve aspect ratio.

    Args:
        width: Image width
        height: Image height
        base_resolution: Base resolution (e.g., 1024)
        bucket_step: Step size for buckets (e.g., 64)

    Returns:
        Tuple of (bucket_width, bucket_height)
    """
    aspect_ratio = width / height
    total_pixels = base_resolution * base_resolution

    # Calculate dimensions that maintain aspect ratio and target pixel count
    bucket_height = int(math.sqrt(total_pixels / aspect_ratio))
    bucket_width = int(bucket_height * aspect_ratio)

    # Round to nearest bucket step
    bucket_width = round(bucket_width / bucket_step) * bucket_step
    bucket_height = round(bucket_height / bucket_step) * bucket_step

    # Ensure minimum size
    bucket_width = max(bucket_width, bucket_step)
    bucket_height = max(bucket_height, bucket_step)

    # Clamp to reasonable max (2x base resolution)
    max_res = base_resolution * 2
    bucket_width = min(bucket_width, max_res)
    bucket_height = min(bucket_height, max_res)

    return bucket_width, bucket_height


class LoRADataset(Dataset):
    """Dataset for FLUX LoRA fine-tuning with dual tokenizers and bucketing."""

    def __init__(
        self,
        data_dir: str,
        tokenizer,  # CLIP tokenizer
        tokenizer_2,  # T5 tokenizer
        resolution: int = 1024,
        caption_ext: str = ".txt",
        use_bucketing: bool = True,
        bucket_step: int = 64
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.resolution = resolution
        self.caption_ext = caption_ext
        self.use_bucketing = use_bucketing
        self.bucket_step = bucket_step

        # Find all images
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            self.image_paths.extend(self.data_dir.glob(f"*{ext}"))
            self.image_paths.extend(self.data_dir.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(self.image_paths)} images in {data_dir}")

        # Cache image sizes for bucketing
        if self.use_bucketing:
            logger.info("Computing bucket assignments for aspect ratio preservation...")
            self.image_buckets = {}
            for img_path in self.image_paths:
                with Image.open(img_path) as img:
                    bucket_w, bucket_h = get_bucket_for_image(
                        img.width, img.height, self.resolution, self.bucket_step
                    )
                    self.image_buckets[str(img_path)] = (bucket_w, bucket_h)
            logger.info(f"Bucketing enabled: {len(set(self.image_buckets.values()))} unique buckets")

        # Base transforms (normalization only)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path = self.image_paths[idx]

        # Load image
        image = Image.open(image_path)
        if image.mode != "RGB":
            # Convert RGBA to RGB with white background
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert("RGB")

        # Apply bucketed resize or standard resize
        if self.use_bucketing:
            bucket_w, bucket_h = self.image_buckets[str(image_path)]
            # Resize to bucket size while maintaining aspect ratio
            image = image.resize((bucket_w, bucket_h), Image.Resampling.LANCZOS)
        else:
            # Standard center crop
            image = transforms.Resize(
                self.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            )(image)
            image = transforms.CenterCrop(self.resolution)(image)

        # Apply normalization
        pixel_values = self.normalize(image)

        # Load caption
        caption = self._load_caption(image_path)

        # Tokenize with CLIP tokenizer
        tokenized_clip = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize with T5 tokenizer (longer max length)
        tokenized_t5 = self.tokenizer_2(
            caption,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized_clip.input_ids.squeeze(0),  # CLIP tokens
            "input_ids_2": tokenized_t5.input_ids.squeeze(0),  # T5 tokens
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
