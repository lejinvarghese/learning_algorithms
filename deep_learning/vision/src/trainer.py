"""
Unified LoRA trainer for FLUX/SD models.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import wandb

from transformers import (
    AutoTokenizer, 
    CLIPTextModel,
    CLIPTextModelWithProjection
)
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    FluxPipeline,
    StableDiffusion3Pipeline,
    DDPMScheduler
)
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb

from src.config import Config
from src.dataset import LoRADataset

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Unified trainer for FLUX/SD LoRA fine-tuning."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.tokenizer = None
        self.text_encoder = None
        self.unet = None
        self.vae = None
        self.noise_scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def setup_model(self):
        """Load and setup model components."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        if "flux" in self.config.model_name.lower():
            self._setup_flux()
        else:
            self._setup_stable_diffusion()
            
        self._setup_lora()
        
    def _setup_flux(self):
        """Setup FLUX model components."""
        # Load FLUX pipeline
        pipe = FluxPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
            variant="fp16" if self.config.mixed_precision == "fp16" else None
        )
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.transformer  # FLUX uses transformer
        self.vae = pipe.vae
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.model_name, 
            subfolder="scheduler"
        )
        
    def _setup_stable_diffusion(self):
        """Setup Stable Diffusion model components."""
        # Load SD3.5 pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
            variant="fp16" if self.config.mixed_precision == "fp16" else None
        )
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.transformer  # SD3.5 also uses transformer
        self.vae = pipe.vae
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.model_name,
            subfolder="scheduler"
        )
        
    def _setup_lora(self):
        """Setup LoRA configuration."""
        # Define target modules based on model type
        if "flux" in self.config.model_name.lower():
            target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ]
        else:  # SD3.5
            target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ]
            
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.DIFFUSION,
            use_rslora=True  # Enable rsLoRA
        )
        
        # Apply LoRA to UNet/Transformer
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Move to device
        self.unet.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        
        # Enable training mode only for LoRA parameters
        self.unet.train()
        self.text_encoder.eval()
        self.vae.eval()
        
        # Freeze non-LoRA parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False
            
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters (only LoRA)
        trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
        
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Use AdamW8bit for memory efficiency
        self.optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Simple cosine scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs * 100,  # Approximate steps
            eta_min=self.config.learning_rate * 0.1
        )
        
    def setup_dataloader(self) -> DataLoader:
        """Setup training dataloader."""
        dataset = LoRADataset(
            data_dir=self.config.data_dir,
            tokenizer=self.tokenizer,
            resolution=self.config.resolution
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss with Min-SNR weighting."""
        # Get batch data
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
            
        # Predict noise
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]
        
        # Compute loss
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        
        # Apply Min-SNR weighting (2025 improvement)
        snr = self._compute_snr(timesteps)
        min_snr_gamma = 5.0
        snr_weight = torch.stack([snr, torch.full_like(snr, min_snr_gamma)]).min(dim=0)[0] / snr
        loss = loss * snr_weight
        
        return loss.mean()
        
    def _compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio for Min-SNR weighting."""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
        sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        
        # SNR = alpha^2 / (1 - alpha^2)
        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return snr
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.unet.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Forward pass
            loss = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
            # Logging
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # W&B logging
            if self.config.use_wandb and step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/epoch": self.epoch,
                    "train/step": self.global_step
                })
                
        return total_loss / num_batches
        
    def save_checkpoint(self, save_path: str):
        """Save LoRA checkpoint."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save LoRA adapter
        self.unet.save_pretrained(save_path)
        
        # Save config
        config_path = os.path.join(save_path, "training_config.yaml")
        self.config.save_yaml(config_path)
        
        logger.info(f"Saved checkpoint to {save_path}")
        
    def train(self):
        """Main training loop."""
        logger.info("Starting LoRA training...")
        
        # Setup
        self.setup_model()
        self.setup_optimizer()
        dataloader = self.setup_dataloader()
        
        # W&B setup
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.project_name,
                config=self.config.__dict__
            )
            
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            avg_loss = self.train_epoch(dataloader)
            
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.config.output_dir, 
                "checkpoints", 
                f"epoch_{epoch + 1}"
            )
            self.save_checkpoint(checkpoint_path)
            
        # Save final checkpoint
        final_path = os.path.join(self.config.output_dir, "checkpoints", "final")
        self.save_checkpoint(final_path)
        
        logger.info("Training completed!")
        
        if self.config.use_wandb:
            wandb.finish()