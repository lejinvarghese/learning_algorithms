import torch
from datetime import datetime
import click
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, ClassLabel
from torch.utils.data import DataLoader
from accelerate import Accelerator
import deepspeed

def to_lowercase(examples):
    examples["query"] = [q.lower() for q in examples["query"]]
    examples["title"] = [s.lower() for s in examples["title"]]
    return examples


@click.command()
@click.option("--model_id", default="answerdotai/ModernBERT-base", help="Model ID")
@click.option("--n_epochs", default=10, help="Number of epochs")
@click.option("--batch_size", default=16, help="Batch size")
def main(model_id, n_epochs, batch_size):
    ## Load dataset and tokenizer
    run_id = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{model_id}'
    dataset = load_dataset("dair-ai/emotion", "unsplit", split="train", columns=["text", "label"])
    dataset = dataset.map(to_lowercase, batched=True, num_proc=12)
    unique_labels = list(set(dataset["query"]))
    n_classes = len(unique_labels)
    labels = ClassLabel(names=unique_labels)
    dataset = dataset.cast_column("query", ClassLabel(names=labels))

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def collate_fn(batch):
        texts = [x["title"] for x in batch]
        labels = torch.tensor([x["query"] for x in batch])
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return encodings.input_ids, encodings.attention_mask, labels

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=12, pin_memory=True, shuffle=True)

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
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 5e-5,
                "warmup_num_steps": 50,
            },
        },
    }

    model, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())
    dataloader = accelerator.prepare(dataloader)

    # Training loop
    model.train()
    for epoch in range(n_epochs):
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            colour="green",
        )

        total_loss = 0.0
        for _, batch in enumerate(progress_bar):
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss = 0.9 * total_loss + 0.1 * loss.item()

            model.backward(loss)
            model.step()
            progress_bar.set_description(f"Epoch {epoch} | Loss: {total_loss:.4f}")

        if accelerator.is_local_main_process:
            click.secho(f"Epoch {epoch} completed | Final loss: {total_loss:.4f}", fg="yellow")
            if epoch % 5 == 0:
                model_path = f"models/{run_id}/checkpoint_{epoch}"
                model.save_checkpoint(model_path)
                click.secho(f"Saved checkpoint to {model_path}", fg="blue")

    if accelerator.is_local_main_process:
        model_path = f"models/{run_id}/final_model"
        model.module.save_pretrained(
            model_path,
            save_config=True,
            safe_serialization=True,
        )
        tokenizer.save_pretrained("final_model")

        click.secho("Training complete and model saved!", fg="green")


if __name__ == "__main__":
    main()
