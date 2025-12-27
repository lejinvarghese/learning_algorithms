"""
Data processing pipeline with aesthetic-focused captioning.
"""

import os
from pathlib import Path
from PIL import Image
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

from src.config import Config

logger = logging.getLogger(__name__)


# Aesthetic trigger words for style learning
AESTHETIC_TRIGGERS = [
    "aesthetic style",
    "artistic composition",
    "visual aesthetic",
    "aesthetic artwork",
    "stylized art"
]


class DataProcessor:
    """Data processing for images with aesthetic-focused captions."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load BLIP for captioning
        self.blip_processor = None
        self.blip_model = None
        self.caption_counter = 0
        
    def setup_captioning(self):
        """Setup BLIP for automatic captioning."""
        logger.info("Loading BLIP for captioning...")
        
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
    def process_image(self, image_path: Path, output_dir: Path) -> bool:
        """Process a single image."""
        try:
            # Load and convert image
            image = Image.open(image_path)
            
            # Convert RGBA to RGB if needed
            if image.mode == "RGBA":
                # Create white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
                
            # Resize if too large
            max_size = self.config.resolution * 2  # Allow up to 2x target resolution
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
            # Save processed image
            output_path = output_dir / image_path.name
            image.save(output_path, "JPEG", quality=95)
            
            # Generate aesthetic-focused caption
            if self.blip_model is not None:
                base_caption = self._generate_caption(image)
                # Add aesthetic trigger to help LoRA learn style
                trigger = AESTHETIC_TRIGGERS[self.caption_counter % len(AESTHETIC_TRIGGERS)]
                caption = f"{trigger}, {base_caption}"
                self.caption_counter += 1
            else:
                # Simple trigger-only caption if no BLIP
                caption = AESTHETIC_TRIGGERS[self.caption_counter % len(AESTHETIC_TRIGGERS)]
                self.caption_counter += 1

            caption_path = output_path.with_suffix(".txt")
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return False
            
    def _generate_caption(self, image: Image.Image) -> str:
        """Generate caption using BLIP."""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.blip_model.generate(**inputs, max_length=50)
                caption = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
                
            return caption
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "an image"
            
    def process_directory(self, input_dir: str, output_dir: str) -> int:
        """Process all images in directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        logger.info(f"Found {len(image_files)} images to process")
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return 0
            
        # Setup captioning if needed
        caption_mode = getattr(self.config, 'caption_mode', 'trigger')
        if caption_mode == 'blip':
            self.setup_captioning()
        elif caption_mode == 'trigger':
            logger.info("Using aesthetic trigger words only (no BLIP)")
        else:
            logger.warning(f"Unknown caption_mode: {caption_mode}, defaulting to trigger-only")
            
        # Process images
        processed = 0
        for image_path in image_files:
            if self.process_image(image_path, output_path):
                processed += 1
                logger.info(f"Processed: {image_path.name}")
                
        logger.info(f"Processed {processed}/{len(image_files)} images")
        return processed


async def process_data(config: Config) -> int:
    """Convenience function to process scraped data."""
    processor = DataProcessor(config)
    
    # Process images from scraper output to training data
    raw_dir = os.path.join(config.data_dir, "raw")
    train_dir = os.path.join(config.data_dir, "train")
    
    if not os.path.exists(raw_dir):
        logger.error(f"Raw data directory not found: {raw_dir}")
        return 0
        
    return processor.process_directory(raw_dir, train_dir)