import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class TripletDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.triplets = []

        # Filter out entries with 'neutral' labels and missing entries
        filtered_data = [
            (entry["premise"], entry["hypothesis"], entry["label"])
            for entry in dataset
            if entry["label"] != -1 and entry["premise"] and entry["hypothesis"]
        ]

        # Create triplets: (anchor, positive, negative)
        # For each sentence pair with "entailment" label, find a negative example
        entailment_pairs = [
            (p, h) for p, h, label in filtered_data if label == 0
        ]  # 0 is entailment in SNLI
        contradiction_pairs = [
            (p, h) for p, h, label in filtered_data if label == 2
        ]  # 2 is contradiction in SNLI

        # Create triplets (anchor, positive, negative)
        for premise, hypothesis in entailment_pairs[
            :10_000_000
        ]:  # Limit for memory reasons
            # The anchor is the premise
            anchor = premise
            # The positive is the entailed hypothesis
            positive = hypothesis
            # Find a random contradiction as negative
            if contradiction_pairs:
                neg_premise, neg_hypothesis = random.choice(contradiction_pairs)
                negative = neg_hypothesis
                self.triplets.append((anchor, positive, negative))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        # Tokenize all three sentences
        anchor_encoding = self.tokenizer(
            anchor,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        positive_encoding = self.tokenizer(
            positive,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        negative_encoding = self.tokenizer(
            negative,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Return tensors of input_ids, attention_mask
        return {
            "anchor_input_ids": anchor_encoding["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor_encoding["attention_mask"].squeeze(0),
            "positive_input_ids": positive_encoding["input_ids"].squeeze(0),
            "positive_attention_mask": positive_encoding["attention_mask"].squeeze(0),
            "negative_input_ids": negative_encoding["input_ids"].squeeze(0),
            "negative_attention_mask": negative_encoding["attention_mask"].squeeze(0),
        }


class TripletDataLoader:
    def __init__(self, tokenizer, repo_id="snli", batch_size=32):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(repo_id)
        self.batch_size = batch_size

    def load(self, split="train"):
        shuffle = True if split == "train" else False
        triplets = TripletDataset(self.dataset[split], self.tokenizer)
        return DataLoader(triplets, batch_size=self.batch_size, shuffle=shuffle)
