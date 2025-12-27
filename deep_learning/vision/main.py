#!/usr/bin/env python3
"""
Simple main entry point for LoRA training.
"""

import asyncio
import argparse
import logging
import oswa

from src.config import Config
from src.trainer import LoRATrainer
from src.scraper import scrape_deviantart
from src.processor import process_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_scrape(config: Config):
    """Run DeviantArt scraping."""
    logger.info("Starting DeviantArt scraping...")
    
    # Create raw data directory
    raw_dir = os.path.join(config.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Update config to point to raw directory
    config.data_dir = raw_dir
    
    count = await scrape_deviantart(config)
    logger.info(f"Scraped {count} images")
    
    return count


async def run_process(config: Config):
    """Run data processing."""
    logger.info("Starting data processing...")
    
    count = await process_data(config)
    logger.info(f"Processed {count} images")
    
    return count


def run_train(config: Config):
    """Run LoRA training."""
    logger.info("Starting LoRA training...")
    
    # Update data_dir to point to training data
    config.data_dir = os.path.join(config.data_dir, "train")
    
    trainer = LoRATrainer(config)
    trainer.train()
    
    logger.info("Training completed!")


async def run_full_pipeline(config: Config):
    """Run complete pipeline: scrape -> process -> train."""
    logger.info("Starting full pipeline...")
    
    # Step 1: Scrape
    scraped = await run_scrape(config)
    if scraped == 0:
        logger.error("No images scraped. Exiting.")
        return
        
    # Step 2: Process
    # Reset data_dir for processing
    config.data_dir = "data"
    processed = await run_process(config)
    if processed == 0:
        logger.error("No images processed. Exiting.")
        return
        
    # Step 3: Train
    run_train(config)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LoRA Personal Taste Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        choices=["scrape", "process", "train", "full"],
        default="full",
        help="Mode to run: scrape, process, train, or full pipeline"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load config
    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = Config()
        logger.info("Using default config")
        
    # Ensure output directories exist
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    # Run based on mode
    if args.mode == "scrape":
        asyncio.run(run_scrape(config))
    elif args.mode == "process":
        asyncio.run(run_process(config))
    elif args.mode == "train":
        run_train(config)
    elif args.mode == "full":
        asyncio.run(run_full_pipeline(config))


if __name__ == "__main__":
    main()