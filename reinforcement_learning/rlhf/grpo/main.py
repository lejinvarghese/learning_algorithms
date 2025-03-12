import numpy as np
import random
import torch
import os
import hashlib
import tarfile
import requests
import re
import wandb
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    PreTrainedTokenizerBase,
)
from trl import GRPOConfig, GRPOTrainer


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.
    Returns None if no valid answer is found.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None

    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None

    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer


def extract_answer_from_dataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def prepare_dataset(split="train"):
    """Load and prepare the GSM8K dataset for training."""
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data = []

    for example in data:
        formatted_example = {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": example["question"]}],
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)

    return formatted_data


def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    return "\n".join([msg["content"].strip() for msg in messages])


class ChatDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        inputs = []
        labels = []
        for example in batch:
            # Here we assume the last message is the target (assistant's output)
            prompt = build_prompt(example["messages"][:-1])
            target = example["messages"][-1]["content"]

            # Concatenate prompt and target (add a newline between them)
            full_text = prompt + "\n" + target
            tokenized = self.tokenizer(full_text, truncation=True, max_length=self.max_length)
            input_ids = torch.tensor(tokenized["input_ids"])
            inputs.append(input_ids)
            # You can choose to set labels equal to input_ids, or modify as needed.
            labels.append(input_ids)

        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": inputs_padded, "labels": labels_padded}


def download_and_extract_cot_archive(url, extract_path):
    """Download and extract the CoT archive if not already done."""
    archive_path = os.path.join(extract_path, "cot.tar.gz")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    if not os.path.exists(archive_path):
        print("Downloading CoT archive...")
        r = requests.get(url, stream=True)
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    # Extract the archive if not already extracted.
    extract_dir = os.path.join(extract_path, "cot_files")
    if not os.path.exists(extract_dir):
        print("Extracting CoT archive...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
    return extract_dir
