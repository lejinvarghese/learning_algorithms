import os
import json
import logging
from tqdm import tqdm
from warnings import filterwarnings

import torch
from transformers import get_cosine_schedule_with_warmup
from gliner import GLiNER

from constants import DIRECTORY

filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

train_path = f"{DIRECTORY}/train_ner.json"

with open(train_path, "r") as f:
    data = json.load(f)

valid_path = f"{DIRECTORY}/valid_ner.json"
with open(valid_path, "r") as f:
    valid = json.load(f)

model = GLiNER.from_pretrained("urchade/gliner_small")

from types import SimpleNamespace

# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=1000,  # number of training iteration
    train_batch_size=4,
    eval_every=100,  # evaluation/saving steps
    save_directory="logs",  # where to save checkpoints
    warmup_ratio=0.1,  # warmup steps
    device="cuda" if torch.cuda.is_available() else "cpu",
    lr_encoder=1e-6,  # learning rate for the backbone
    lr_others=5e-5,  # learning rate for other parameters
    freeze_token_rep=False,  # freeze of not the backbone
    # Parameters for set_sampling_params
    max_types=25,  # maximum number of entity types during training
    shuffle_types=True,  # if shuffle or not entity types
    random_drop=True,  # randomly drop entity types
    max_neg_type_ratio=2,  # ratio of positive/negative types, 1 mean 50%/50%, 2 mean 33%/66%, 3 mean 25%/75% ...
    max_len=384,  # maximum sentence length
)


def train(model, config, train_data, eval_data=None):
    model = model.to(config.device)

    # Set sampling parameters from config
    model.set_sampling_params(
        max_types=config.max_types,
        shuffle_types=config.shuffle_types,
        random_drop=config.random_drop,
        max_neg_type_ratio=config.max_neg_type_ratio,
        max_len=config.max_len,
    )

    model.train()

    # Initialize data loaders
    train_loader = model.create_dataloader(
        train_data, batch_size=config.train_batch_size, shuffle=True
    )

    # Optimizer
    optimizer = model.get_optimizer(
        config.lr_encoder, config.lr_others, config.freeze_token_rep
    )

    pbar = tqdm(range(config.num_steps))

    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps,
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(config.device)

        loss = model(x)  # Forward pass

        # Check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
        pbar.set_description(description)

        if (step + 1) % config.eval_every == 0:

            model.eval()

            if eval_data is not None:
                results, f1 = model.evaluate(
                    eval_data["samples"],
                    flat_ner=True,
                    threshold=0.5,
                    batch_size=12,
                    entity_types=eval_data["entity_types"],
                )

                print(f"Step={step}\n{results}")

            if not os.path.exists(config.save_directory):
                os.makedirs(config.save_directory)

            model.save_pretrained(f"{config.save_directory}/finetuned_{step}")

            model.train()


eval_data = {
    "entity_types": [
        # "product",
        # "product taxonomy",
        # "product category",
        # "sku",
        "brand",
        # "organization",
        # "person",
    ],
    "samples": data[:50],
}

train(model, config, data, eval_data)
