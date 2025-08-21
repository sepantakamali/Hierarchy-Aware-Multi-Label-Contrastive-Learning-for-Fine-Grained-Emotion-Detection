import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import ast


# Utility function for converting labels from comma separated strings to a multi-label list
def parse_labels(row):
    if isinstance(row, list):
        return row
    if isinstance(row, str):
        try:
            parsed = ast.literal_eval(row)
            if isinstance(parsed, list):
                print(f"[parse_labels] Parsed labels: {parsed}")
                return [label.strip() for label in parsed]
        except Exception:
            parsed = [label.strip() for label in row.split(",")]
            # print(f"[parse_labels] Parsed labels: {parsed}")
            return parsed
    return []

class MultiLabelContrastiveDataset(Dataset):
    def __init__(self, csv_path, label_list=None, tokenizer_name="joeddav/distilbert-base-uncased-go-emotions-student", max_length=128, neutral_only=False, split="train"):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.neutral_only = neutral_only
        self.split = split

        # Parse multi-label strings into lists using robust parse_labels function
        self.data["fine"] = self.data["fine"].apply(parse_labels)

        # Enforce neutral isolation in label sets
        if self.neutral_only:
            self.data["fine"] = [["neutral"] for _ in range(len(self.data))]
            for labels in self.data["fine"]:
                assert labels == ["neutral"], "Neutral-only dataset contains non-neutral labels"
        elif split == "train":
            # Remove 'neutral' from mixed labels only during training
            for i, labels in enumerate(self.data["fine"]):
                if "neutral" in labels:
                    self.data.at[i, "fine"] = [label for label in labels if label != "neutral"]
            # Drop samples that end up with no labels
            self.data = self.data[self.data["fine"].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        # Collect existing labels from data if label_list is not provided, always include "neutral"
        if label_list is None:
            unique_labels = list({label for row in self.data["fine"] for label in row if label != "neutral"})
            collected_labels = sorted(unique_labels) + ["neutral"]  
        else:
            collected_labels = label_list  # Preserve the original order and avoid sorting again

        if label_list is not None and "neutral" not in label_list and any("neutral" in labels for labels in self.data["fine"]):
            raise ValueError("'neutral' found in data but not in provided label_list.")

        self.label_list = collected_labels
        self.mlb = MultiLabelBinarizer(classes=self.label_list) # Multi hot label encoder
        self.mlb.fit([self.label_list])  # Fit once

        print(f"[INIT] Loaded {split} dataset from {csv_path} | Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Retrieve Row, Encode, Construct Dataset structure"""
        
        row = self.data.iloc[index]
        text = str(row["text"])
        labels = row["fine"]

        # Encoder setup
        encode = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" # Return as PyTorch tensors (batch dimension included)
        )

        multi_hot_encoded = self.mlb.transform([labels])[0] # Multi hot encoded labels list

        # Validation: enforce encoding length and isolation
        assert len(multi_hot_encoded) == len(self.label_list), "Label encoding size mismatch"
        if self.neutral_only:
            assert multi_hot_encoded[:-1].sum() == 0, "Neutral-only instance has non-neutral label active"
        elif self.split == "train" and not self.neutral_only and multi_hot_encoded[-1] == 1:
            print(f"[DEBUG] Unexpected 'neutral' in training data at index {index} | Labels: {labels} | Encoded: {multi_hot_encoded}")
            assert multi_hot_encoded[-1] != 1, "'neutral' label should not be active in training data"
        
        item = {
            "input_ids": encode["input_ids"].squeeze(0), # Encoded text--token ids to feed to the model
            "attention_mask": encode["attention_mask"].squeeze(0), # 1 for real tokens, 0 for padding
            "labels": torch.FloatTensor(multi_hot_encoded), # Multi-hot vector (target labels)
            "raw_labels": labels, # Original string labels (used for contrastive pair mining) 
            "text": text # For visualisation, debugging, etc.
        }
        
        # print(f"[__getitem__] Index {index} | Labels: {labels} | Encoded: {multi_hot_encoded.tolist()}")
        return item