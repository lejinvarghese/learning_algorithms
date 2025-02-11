import os
import logging
from datetime import datetime
import click
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
import deepspeed

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.utils.backcompat.broadcast_warning.enabled = False  # Suppress specific warnings
logging.getLogger("torch").setLevel(logging.ERROR)
class_names = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
class_names = {v: k for k, v in class_names.items()}


@click.command()
@click.option("--model_id", default="answerdotai/ModernBERT-base", help="Model ID")
@click.option("--n_epochs", default=10, help="Number of epochs")
@click.option("--batch_size", default=16, help="Batch size")
def main(model_id, n_epochs, batch_size):
    ## Load dataset and tokenizer
    run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{model_id}'
    dataset = load_dataset("dair-ai/emotion", "unsplit", split="train", columns=["text", "label"])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    unique_labels = list(set(dataset["train"]["label"]))
    n_classes = len(unique_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch])
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return encodings.input_ids, encodings.attention_mask, labels

    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True,
    )

    ## Load model
    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=n_classes)

    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": batch_size,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu"},
            "offload_optimizer": {"device": "cpu"},
            "stage3_param_persistence_threshold": 10000,
            "stage3_max_live_parameters": 0,
            "stage3_prefetch_bucket_size": 0,
            "memory_efficient_linear": False,
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 50,
            },
        },
    }

    model, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())
    train_dataloader = accelerator.prepare(train_dataloader)
    test_dataloader = accelerator.prepare(test_dataloader)
    writer = SummaryWriter(log_dir=f"logs/{run_id}")
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            colour="yellow",
        )

        total_loss = 0.0
        for step, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss = 0.9 * total_loss + 0.1 * loss.item()

            model.backward(loss)
            model.step()
            progress_bar.set_description(f"Epoch {epoch} | Loss: {total_loss:.4f}")
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_dataloader) + step)

        if accelerator.is_local_main_process:
            click.secho(f"Epoch {epoch} completed | Final loss: {total_loss:.4f}", fg="yellow")
            writer.add_scalar("Loss/epoch", total_loss, epoch)

            if epoch % 5 == 0:
                ## save
                model_path = f"models/{run_id}/checkpoint_{epoch}"
                model.save_checkpoint(model_path)
                click.secho(f"Saved checkpoint to {model_path}", fg="blue")

        ## evaluate
        ## qualitative evaluation on sample sentences
        model.eval()
        sample_texts = ["i'm a happy camper", "this is so frustrating", "im alright"]
        encodings = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        predictions = [class_names[p] for p in predictions]
        click.secho(f"Sample Predictions: {list(zip(sample_texts, predictions))}", fg="magenta")

        ## quantitative evaluation on sample sentences
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluating", colour="blue"):
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
        writer.add_scalar("Accuracy/test", accuracy, 0)
        writer.add_scalar("Precision/test", precision, 0)
        writer.add_scalar("Recall/test", recall, 0)
        writer.add_scalar("F1/test", f1, 0)

    if accelerator.is_local_main_process:
        model_path = f"models/{run_id}/final_model"
        model.module.save_pretrained(
            model_path,
            save_config=True,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(model_path)

        click.secho("Training complete and model saved!", fg="green")


if __name__ == "__main__":
    main()
