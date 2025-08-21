import pandas as pd
import os
from pathlib import Path
import json

# ///////////////////////////// Creating dataset based on official splits //////////////////////////// #

# Load emotion index-to-label mapping
with open("data/goemotions_splits/emotions.txt", "r", encoding="utf8") as f:
    EMOTIONS = [line.strip() for line in f.readlines()]

# Define hierarchy mapping: fine → intermediate → coarse
HIERARCHY = {
    # Positive → Joy
    "admiration": ("joy", "positive"),
    "amusement": ("joy", "positive"),
    "approval": ("joy", "positive"),
    "caring": ("joy", "positive"),
    "desire": ("joy", "positive"),
    "excitement": ("joy", "positive"),
    "gratitude": ("joy", "positive"),
    "joy": ("joy", "positive"),
    "love": ("joy", "positive"),
    "optimism": ("joy", "positive"),
    "pride": ("joy", "positive"),
    "relief": ("joy", "positive"),

    # Negative → Anger
    "anger": ("anger", "negative"),
    "annoyance": ("anger", "negative"),
    "disapproval": ("anger", "negative"),

    # Negative → Disgust
    "disgust": ("disgust", "negative"),

    # Negative → Fear
    "fear": ("fear", "negative"),
    "nervousness": ("fear", "negative"),

    # Negative → Sadness
    "sadness": ("sadness", "negative"),
    "disappointment": ("sadness", "negative"),
    "embarrassment": ("sadness", "negative"),
    "grief": ("sadness", "negative"),
    "remorse": ("sadness", "negative"),

    # Ambiguous → Surprise
    "confusion": ("surprise", "ambiguous"),
    "curiosity": ("surprise", "ambiguous"),
    "realization": ("surprise", "ambiguous"),
    "surprise": ("surprise", "ambiguous"),

    # Neutral
    "neutral": ("neutral", "neutral"),
}

# Sanity check: all EMOTIONS must have hierarchy mappings
missing = [label for label in EMOTIONS if label not in HIERARCHY]
assert not missing, f"The following labels are missing from HIERARCHY: {missing}"

# Load a split and map labels
def load_file(path, split):
    dataframe = pd.read_csv(path, sep="\t", names=["text", "labels", "id"])

    # print("PATH:", path)
    # print("COLUMNS:", dataframe.columns.tolist())
    # print(dataframe.head(5))

    data = []
    for _, row in dataframe.iterrows():
        # print(row["text"] + "-->" + row["labels"])
        label_str = str(row["labels"]).strip("[]") # Remove brakets
        indices = [int(index) for index in label_str.split(",") if index.strip().isdigit()]
        
        if not indices:
            continue

        fine = [EMOTIONS[index] for index in indices]
        intermediate = list(set(HIERARCHY[label][0] for label in fine))
        coarse = list(set(HIERARCHY[label][1] for label in fine))

        data.append({
            "text": row["text"],
            "fine": fine,
            "intermediate": intermediate,
            "coarse": coarse,
            "split": split
        })
    return data

# Save JSONL
def save_as_jsonl(data, destination):
    with open(destination, "w", encoding="utf8") as file:
        for record in data:
            file.write(json.dumps(record) + "\n")

# Save CSV
def save_as_csv(data, destination):
    flat_data = []
    for record in data:
        flat_data.append({
            "text": record["text"],
            "fine": ", ".join(record["fine"]),
            "intermediate": ", ".join(record["intermediate"]),
            "coarse": ", ".join(record["coarse"]),
            "split": record["split"]
        })
    dataframe = pd.DataFrame(flat_data)
    dataframe.to_csv(destination, index=False, encoding="utf8")

# Load the splitted data files
def load_goemotions_official(data_directory="data/goemotions_splits"):
    files = [("train.tsv", "train"), ("dev.tsv", "dev"), ("test.tsv", "test")]
    all_data = [] # For generating an acculmulated version for analysis
    os.makedirs("data/extracted", exist_ok=True)

    for filename, split in files:
        path = Path(data_directory) / filename
        data = load_file(path, split)
        save_as_jsonl(data, f"data/extracted/goemotions_{split}.jsonl") # Splits
        save_as_csv(data, f"data/extracted/goemotions_{split}.csv") # Splits
        all_data.extend(data)

    return all_data

# Run as script
if __name__ == "__main__":
    data = load_goemotions_official("data/goemotions_splits")
    # Create directory if doesn't exist
    os.makedirs("data/extracted", exist_ok=True)
    # Save JSONL
    save_as_jsonl(data, "data/extracted/goemotions_official.jsonl") # Accumlated
    # Save CSV
    save_as_csv(data, "data/extracted/goemotions_official.csv") # Accumlated
    # Approval Message
    print(f"Extracted {len(data)} records and saved to data/extracted/... in JSONL and CSV formats.")