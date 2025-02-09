import torch
import click
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
import deepspeed


def collate_fn(batch):
    texts = [x["sentence"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch])
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return encodings.input_ids, encodings.attention_mask, labels


accelerator = Accelerator()
dataset = load_dataset("glue", "sst2", split="train[:1%]")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# DeepSpeed configuration
ds_config = {
    "train_batch_size": 8,  # Match your DataLoader batch size
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
        "params": {"warmup_min_lr": 0, "warmup_max_lr": 5e-5, "warmup_num_steps": 50},
    },
}


model, optimizer, _, _ = deepspeed.initialize(
    model=model, config=ds_config, model_parameters=model.parameters()
)
dataloader = accelerator.prepare(dataloader)

# Training loop
model.train()
for epoch in range(10):
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=not accelerator.is_local_main_process,
        colour="green",
    )

    total_loss = 0.0
    for batch_idx, batch in enumerate(progress_bar):
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
