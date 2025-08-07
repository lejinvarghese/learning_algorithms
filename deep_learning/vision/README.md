# LoRA Personal Taste Training Pipeline

A modern, production-ready pipeline for training LoRA (Low-Rank Adaptation) checkpoints that capture your personal artistic taste from DeviantArt favorites. Built with 2025's best practices and latest Stable Diffusion models.

## 🌟 Features

- **Latest Models**: Support for Stable Diffusion 3.5, FLUX.1, and SDXL
- **Advanced LoRA**: rsLoRA, optimized hyperparameters, Min-SNR weighting
- **Smart Data Pipeline**: DeviantArt scraping, automatic captioning, bucketing
- **Cloud Deployment**: SkyPilot integration for scalable GCP training
- **Modern Architecture**: Modular, async, type-safe Python codebase
- **Comprehensive Monitoring**: W&B integration, TensorBoard, sample generation

## 🚀 Quick Start

### 1. Setup

```bash
# Clone and install
git clone <your-repo>
cd vision
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Create default config
python scripts/create_default_config.py

# Edit config.yaml with your settings:
# - DeviantArt favorites URL
# - API credentials  
# - Training parameters
```

### 3. Local Training

```bash
# Run full pipeline (scrape → process → train)
python main.py --mode full

# Or run individual steps
python main.py --mode scrape    # Just scrape images
python main.py --mode process   # Just process data
python main.py --mode train     # Just train model
```

### 4. Cloud Training (Recommended)

```bash
# Install SkyPilot
pip install skypilot-nightly[gcp]

# Setup GCP credentials
sky check

# Launch training on cloud
sky launch sky.yaml

# Monitor progress
sky logs lora-personal-taste --follow
```

## 📖 Detailed Guide

### DeviantArt Setup

1. **Get API Credentials**:
   - Go to [DeviantArt Developers](https://www.deviantart.com/developers/)
   - Create an application
   - Copy Client ID and Secret to `config.yaml`

2. **Configure Scraping**:
   ```yaml
   deviantart:
     favorites_url: "https://www.deviantart.com/USERNAME/favourites"
     client_id: "your_client_id"
     client_secret: "your_client_secret"
     max_images: 200
     include_mature: false
   ```

### Model Selection

The pipeline supports multiple model types:

```yaml
model:
  model_type: "stabilityai/stable-diffusion-3.5-large"  # Recommended
  # Alternatives:
  # "stabilityai/stable-diffusion-3.5-medium"
  # "black-forest-labs/FLUX.1-dev"
  # "black-forest-labs/FLUX.1-schnell" 
  # "stabilityai/stable-diffusion-xl-base-1.0"
```

**Model Recommendations**:
- **SD3.5 Large**: Best for artistic flexibility and style capture
- **FLUX.1 Dev**: Best for photorealistic results
- **SDXL**: Good balance of quality and speed

### Training Configuration

Optimized defaults for 2025:

```yaml
model:
  lora_rank: 16              # Balanced efficiency/quality
  lora_alpha: 16.0           # 1:1 ratio for rsLoRA
  use_rslora: true           # Enable rank-stabilized LoRA
  mixed_precision: "bf16"    # Best for modern GPUs

training:
  num_epochs: 3              # Prevent overfitting
  batch_size: 1              # Conservative for high-res
  learning_rate: 1e-5        # 2025 recommended
  min_snr_gamma: 5.0         # Min-SNR weighting
  lr_scheduler: "cosine_with_restarts"
```

### Data Processing

The pipeline automatically:

1. **Scrapes** your DeviantArt favorites
2. **Cleans** and converts images (RGBA→RGB, resize, crop)
3. **Generates captions** using BLIP
4. **Creates tags** using WD14 tagger
5. **Buckets** images by aspect ratio
6. **Caches latents** for efficient training

## 🏗️ Architecture

```
src/
├── core/
│   ├── config.py      # Configuration management
│   └── trainer.py     # Main training loop
├── data/
│   ├── scraper.py     # DeviantArt scraping
│   ├── processor.py   # Data processing pipeline
│   └── dataset.py     # PyTorch dataset
└── models/
    └── stable_diffusion.py  # Model loading & LoRA
```

### Key Components

- **Config System**: Type-safe, validated YAML configuration
- **Async Scraper**: Rate-limited DeviantArt API client
- **Smart Processing**: BLIP captioning, WD14 tagging, bucketing
- **Modern Training**: Min-SNR loss, rsLoRA, gradient checkpointing
- **Cloud Ready**: SkyPilot integration for scalable deployment

## 🎯 Advanced Usage

### Custom Training Data

```bash
# Use your own images (put in data/train/)
python main.py --mode process  # Process existing data
python main.py --mode train    # Train on processed data
```

### Hyperparameter Tuning

```yaml
training:
  learning_rate: 1e-5        # Start here, adjust based on results
  batch_size: 1              # Increase if you have more VRAM
  gradient_accumulation_steps: 4  # Effective batch size = 4
  num_epochs: 3              # More epochs = more overfitting risk
  
model:
  lora_rank: 16              # Higher = more capacity, slower
  lora_alpha: 16.0           # Usually equal to rank for rsLoRA
```

### Multi-GPU Training

The pipeline supports multi-GPU training via PyTorch DDP:

```bash
# Local multi-GPU
torchrun --nproc_per_node=2 main.py --mode train

# Cloud multi-GPU (edit sky.yaml)
accelerators: A100:2
```

### Experiment Tracking

```yaml
training:
  log_with: "wandb"         # or "tensorboard"
  project_name: "my-lora-project"
```

### Model Deployment

After training, use your LoRA with any diffusion interface:

```python
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel

# Load base model
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large")

# Load your trained LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet, "outputs/checkpoints/best")

# Generate with your style
image = pipe("beautiful woman, masterpiece").images[0]
```

## 📊 Monitoring & Debugging

### TensorBoard

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

### Sample Generation

The pipeline automatically generates samples during training:
- `outputs/samples/` - Training progress samples
- `outputs/checkpoints/` - Model checkpoints

### Memory Optimization

For limited VRAM:

```yaml
model:
  gradient_checkpointing: true
  mixed_precision: "fp16"      # Use fp16 instead of bf16
  
training:
  batch_size: 1
  gradient_accumulation_steps: 8  # Simulate larger batches
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `batch_size` to 1
   - Enable `gradient_checkpointing`
   - Use `mixed_precision: "fp16"`

2. **DeviantArt Rate Limiting**:
   - Increase `rate_limit_delay` in config
   - Reduce `max_images`

3. **Model Download Issues**:
   - Ensure you have HuggingFace access to gated models
   - Check internet connection and disk space

4. **Training Instability**:
   - Reduce learning rate
   - Enable Min-SNR weighting
   - Use gradient clipping

### Debug Mode

```bash
python main.py --log-level DEBUG --mode train
```

## 🌐 Cloud Deployment

### SkyPilot Setup

```bash
# Install SkyPilot
pip install skypilot-nightly[gcp]

# Setup cloud credentials
sky check

# Launch training
sky launch sky.yaml

# Check status
sky status

# View logs  
sky logs lora-personal-taste --follow

# Stop when done
sky stop lora-personal-taste
```

### Cost Optimization

- Use spot instances (`use_spot: true`)
- Choose appropriate GPU (V100 vs A100)
- Monitor training and stop early if needed
- Auto-shutdown on completion

### Storage

- Code synced automatically via `file_mounts`
- Outputs saved to persistent disk
- Download results: `sky down lora-personal-taste:~/outputs ./outputs`

## 📈 Performance Tips

### Training Speed
- Use `mixed_precision: "bf16"` on modern GPUs
- Enable `xformers` memory efficient attention
- Use bucketing for variable aspect ratios
- Gradient checkpointing for memory vs speed tradeoff

### Quality Tips
- Start with 50-200 high-quality, diverse images
- Use descriptive filenames (become captions)
- Mix different art styles for versatility  
- Train for 2-3 epochs maximum
- Use Min-SNR weighting for better convergence

### Data Curation
- Remove low-quality or unrelated images
- Ensure good variety in poses, lighting, styles
- Consider mature content filtering
- Balance dataset (avoid too many similar images)

## 🤝 Contributing

We welcome contributions! Please see:
- Issues for bugs and feature requests
- Pull requests for code contributions
- Discussions for questions and ideas

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion models
- Black Forest Labs for FLUX models  
- Hugging Face for diffusers and transformers
- PEFT team for LoRA implementation
- DeviantArt for the API
- SkyPilot for cloud orchestration

---

**Happy Training!** 🎨✨

For questions or support, please open an issue or check the discussions.