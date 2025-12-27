"""
FLUX LoRA trainer using proper diffusers APIs.
Simplified to only support FLUX models.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import logging
from tqdm import tqdm
import wandb
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

from src.config import Config
from src.dataset import LoRADataset

logger = logging.getLogger(__name__)


class LoRATrainer:
    """FLUX LoRA trainer for aesthetic fine-tuning."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.tokenizer = None  # CLIP tokenizer
        self.tokenizer_2 = None  # T5 tokenizer
        self.text_encoder = None  # CLIP-L
        self.text_encoder_2 = None  # T5-XXL
        self.transformer = None
        self.vae = None
        self.noise_scheduler = None
        self.pipeline = None

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Mixed precision
        self.use_amp = config.mixed_precision in ["fp16", "bf16"]
        self.amp_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        self.scaler = GradScaler("cuda") if self.use_amp and config.mixed_precision == "fp16" else None

    def setup_model(self):
        """Load and setup FLUX model components."""
        logger.info(f"Loading FLUX model: {self.config.model_name}")

        # Load FLUX pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=self.amp_dtype if self.use_amp else torch.float32,
        )

        # Extract components - FLUX has dual text encoders
        self.tokenizer = self.pipeline.tokenizer  # CLIP tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2  # T5 tokenizer
        self.text_encoder = self.pipeline.text_encoder  # CLIP-L
        self.text_encoder_2 = self.pipeline.text_encoder_2  # T5-XXL
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae

        # FLUX uses flow matching, not diffusion
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model_name,
            subfolder="scheduler"
        )

        logger.info("Model components loaded successfully")
        self._setup_lora()

    def _setup_lora(self):
        """Setup LoRA configuration for FLUX transformer."""
        logger.info("Setting up LoRA adapters...")

        # FLUX transformer target modules
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",  # Attention
            "ff.net.0.proj", "ff.net.2"  # Feed-forward
        ]

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_rslora=True  # Enable rsLoRA for better initialization
        )

        # Apply LoRA to transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.print_trainable_parameters()

        # Move to device
        self.transformer.to(self.device)
        self.text_encoder.to(self.device)
        self.text_encoder_2.to(self.device)
        self.vae.to(self.device)

        # Set training modes
        self.transformer.train()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.vae.eval()

        # Freeze non-LoRA parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder_2.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

        logger.info("LoRA setup complete")

    def setup_optimizer(self, num_training_steps: int):
        """Setup optimizer and scheduler with accurate step calculation."""
        # Get trainable parameters (only LoRA)
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]

        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {num_trainable:,}")

        # Use AdamW8bit for memory efficiency
        self.optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Cosine scheduler with warmup
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.config.learning_rate * 0.1
        )

        logger.info(f"Optimizer configured for {num_training_steps} training steps")

    def setup_dataloader(self) -> DataLoader:
        """Setup training dataloader."""
        dataset = LoRADataset(
            data_dir=self.config.data_dir,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            resolution=self.config.resolution
        )

        if len(dataset) == 0:
            raise ValueError(f"No images found in {self.config.data_dir}")

        logger.info(f"Dataset size: {len(dataset)} images")

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def encode_prompt(self, input_ids, input_ids_2):
        """Encode prompts using both CLIP and T5 encoders."""
        with torch.no_grad():
            # Encode with CLIP
            prompt_embeds = self.text_encoder(
                input_ids.to(self.device),
                output_hidden_states=False
            )[0]

            # Encode with T5
            prompt_embeds_2 = self.text_encoder_2(
                input_ids_2.to(self.device),
                output_hidden_states=False
            )[0]

            # Get pooled embeddings from CLIP for conditioning
            pooled_prompt_embeds = self.text_encoder(
                input_ids.to(self.device),
                output_hidden_states=False
            )[0]

        return prompt_embeds, prompt_embeds_2, pooled_prompt_embeds

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute training loss for FLUX flow matching."""
        # Get batch data
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"]
        input_ids_2 = batch["input_ids_2"]

        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # Get text embeddings from both encoders
        prompt_embeds, prompt_embeds_2, pooled_prompt_embeds = self.encode_prompt(
            input_ids, input_ids_2
        )

        # Sample noise and timesteps (flow matching uses [0, 1] range)
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]

        # Sample timesteps uniformly for flow matching
        timesteps = torch.rand(batch_size, device=self.device)

        # Flow matching: interpolate between noise and data
        # x_t = (1 - t) * noise + t * latents
        noisy_latents = (1 - timesteps.view(-1, 1, 1, 1)) * noise + timesteps.view(-1, 1, 1, 1) * latents

        # Convert timesteps to scheduler format (multiply by 1000 for FLUX)
        timesteps_scaled = (timesteps * 1000).long()

        # Predict the velocity (target for flow matching is latents - noise)
        target = latents - noise

        # Forward pass through transformer
        model_pred = self.transformer(
            hidden_states=noisy_latents,
            timestep=timesteps_scaled,
            encoder_hidden_states=prompt_embeds_2,  # T5 embeddings go here
            pooled_projections=pooled_prompt_embeds,  # CLIP pooled embeddings
            return_dict=False
        )[0]

        # Compute MSE loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.transformer.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type="cuda", dtype=self.amp_dtype):
                    loss = self.compute_loss(batch)

                # Backward pass with gradient scaling (for fp16 only)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                loss = self.compute_loss(batch)
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
            if self.config.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/epoch": self.epoch,
                    "train/step": self.global_step
                })

        return total_loss / num_batches

    def save_checkpoint(self, save_path: str):
        """Save LoRA checkpoint."""
        os.makedirs(save_path, exist_ok=True)

        # Save LoRA adapter weights
        self.transformer.save_pretrained(save_path)

        # Save config
        config_path = os.path.join(save_path, "training_config.yaml")
        self.config.save_yaml(config_path)

        logger.info(f"Saved checkpoint to {save_path}")

    def generate_validation_images(self, prompts: list[str], save_dir: str):
        """Generate validation images to check LoRA quality."""
        logger.info("Generating validation images...")

        os.makedirs(save_dir, exist_ok=True)

        # Set to eval mode
        self.transformer.eval()

        # Merge LoRA weights temporarily for inference
        self.transformer = self.transformer.merge_and_unload()

        # Update pipeline with trained transformer
        self.pipeline.transformer = self.transformer

        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                    height=self.config.resolution,
                    width=self.config.resolution,
                ).images[0]

                image_path = os.path.join(save_dir, f"val_epoch{self.epoch}_img{i}.png")
                image.save(image_path)

                if self.config.use_wandb:
                    wandb.log({f"validation/image_{i}": wandb.Image(image, caption=prompt)})

        # Re-apply LoRA for continued training
        logger.info("Re-applying LoRA for continued training...")
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "ff.net.0.proj", "ff.net.2"
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_rslora=True
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.to(self.device)
        self.transformer.train()

    def train(self):
        """Main training loop."""
        logger.info("Starting FLUX LoRA training...")

        # Setup
        self.setup_model()
        dataloader = self.setup_dataloader()

        # Calculate total training steps
        num_update_steps_per_epoch = len(dataloader) // self.config.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.config.num_epochs

        self.setup_optimizer(num_training_steps)

        # W&B setup
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.project_name,
                config=self.config.__dict__
            )

        # Validation prompts
        validation_prompts = [
            "a beautiful landscape",
            "a portrait of a person",
            "an abstract artwork"
        ]

        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Train epoch
            avg_loss = self.train_epoch(dataloader)

            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

            # Generate validation images
            if (epoch + 1) % 1 == 0:  # Every epoch
                val_dir = os.path.join(self.config.output_dir, "validation", f"epoch_{epoch + 1}")
                self.generate_validation_images(validation_prompts, val_dir)

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
