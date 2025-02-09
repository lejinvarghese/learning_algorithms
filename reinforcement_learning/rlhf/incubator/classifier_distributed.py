import torch
import click
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import deepspeed


@click.command()
@click.option("--model_id", default="answerdotai/ModernBERT-base", help="Model ID")
@click.option("--n_epochs", default=10, help="Number of epochs")
@click.option("--batch_size", default=16, help="Batch size")
def main(model_id, n_epochs, batch_size):
    ## Load dataset and tokenizer
    dataset = load_dataset("glue", "sst2", split="train[:1%]")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def collate_fn(batch):
        texts = [x["sentence"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch])
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return encodings.input_ids, encodings.attention_mask, labels

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    ## Load model
    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

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

    model, _, _, _ = deepspeed.initialize(
        model=model, config=ds_config, model_parameters=model.parameters()
    )
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

        # Print epoch summary only on main process
        if accelerator.is_local_main_process:
            click.secho(
                f"Epoch {epoch} completed | Final loss: {total_loss:.4f}", fg="yellow"
            )

    if accelerator.is_local_main_process:
        click.secho("Training complete!", fg="green")


if __name__ == "__main__":
    main()
