import pandas as pd
import json
from pathlib import Path
from collections import Counter
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Paths Setup -----
BASE_DIRECTORY = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIRECTORY / "data" / "preprocessed"
OUTPUT_PATH = BASE_DIRECTORY / "data" / "preprocessed"
FIGURES_DIRECTORY = BASE_DIRECTORY / "figures" / "isolated"
FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)


# Utility function for converting labels from comma separeted strings to a multi-label list
def parse_labels(row):
    if isinstance(row, list):
        return row
    if isinstance(row, str):
        try:
            parsed = ast.literal_eval(row)
            if isinstance(parsed, list):
                return [label.strip() for label in parsed] # Return cleaned parsed labels list
        except Exception:
            return [label.strip() for label in row.split(",")] # Return cleaned not parsed labels list
    return []


train = pd.read_csv(INPUT_PATH / "goemotions_train_selected.csv").copy()
neutral = pd.read_csv(INPUT_PATH / "neutral_pool.csv").copy()

train["fine"] = train["fine"].apply(parse_labels)
neutral["fine"] = neutral["fine"].apply(parse_labels)

train_neutrals = train[train["fine"].apply(lambda labels: labels == ["neutral"])]
emotions_isolated = train.drop(index=train_neutrals.index)
neutral_pool = pd.concat([neutral, train_neutrals], ignore_index=True)


# ----- Plotting Utilities -----
def plot_label_frequency(dataframe):
    counter = Counter(label for labels in dataframe["fine"] for label in labels)
    labels, values = zip(*sorted(counter.items(), key=lambda x: x[1], reverse=True))

    plt.figure(figsize=(14, 6))
    sns.barplot(x=list(labels), y=list(values), hue=list(labels), palette="Blues_d", legend=False)
    plt.xticks(rotation=90)
    plt.title(f"Label Frequency Distribution — Train")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    for i, val in enumerate(values):
        plt.text(i, val + max(values) * 0.01, str(val), ha='center', va='bottom', fontsize=6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIRECTORY / f"label_frequency_train.png", dpi=300)
    plt.savefig(FIGURES_DIRECTORY / f"label_frequency_train.pdf")
    plt.close()


plot_label_frequency(emotions_isolated)

emotions_isolated["fine"] = emotions_isolated["fine"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
neutral_pool["fine"] = neutral_pool["fine"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

summary = {
    "original_train_size": len(train),
    "train_neutrals": len(train_neutrals),
    "final_train_size": len(emotions_isolated),
    "pool_size": len(neutral_pool),
    "label_distribution": Counter(label for row in emotions_isolated['fine'] for label in row.split(', '))
}

with open(OUTPUT_PATH / "isolation_summary.json", "w") as file:
    json.dump(summary, file, indent=2)

print(f"\nSummary information saved at {OUTPUT_PATH}\n")

emotions_isolated.to_csv(OUTPUT_PATH / "goemotions_train_isolated.csv", index=False)
neutral_pool.to_csv(OUTPUT_PATH / "neutral_isolated.csv", index=False)

print("Train data and Neutral pool have been updated successfully!\n")
print(f"New Datasets have been saved to: {OUTPUT_PATH}\n")