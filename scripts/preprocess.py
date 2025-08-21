import pandas as pd
import re
import json
import emoji
from pathlib import Path
from collections import Counter
from ekphrasis.classes.preprocessor import TextPreProcessor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="ekphrasis")
from ekphrasis.dicts.emoticons import emoticons as ekphrasis_emoticons
import ast
import statistics
import contractions

# Words limit for text length
MIN_WORDS = 3

# ----- Paths Setup -----
BASE_DIRECTORY = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIRECTORY / "data" / "extracted"
OUTPUT_PATH = BASE_DIRECTORY / "data" / "preprocessed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ----- Load Emoticon Dictionary -----
DICT_PATH = BASE_DIRECTORY / "data" / "emoticons" / "ekphrasis_ready_emoticons_filtered.json"
with open(DICT_PATH, "r", encoding="utf-8") as file:
    custom_emoticons = json.load(file)

# ----- Ekphrasis Mapping to Fine Emotions -----
EKPHRASIS_TAG_MAPPING = {
    "angel": "",
    "annoyed": "<annoyance>",
    "devil": "",
    "happy": "<joy>",
    "heart": "<love>",
    "highfive": "<approval>",
    "kiss": "<love>",
    "laugh": "<joy>",
    "sad": "<sadness>",
    "seallips": "",
    "surprise": "<surprise>",
    "tong": "",
    "wink": ""
}

ekphrasis_filtered = {
    emoticon: EKPHRASIS_TAG_MAPPING[tag[1:-1]] # <1-1> unframe the tag <joy> → joy
    for emoticon, tag in ekphrasis_emoticons.items()
    if tag[1:-1] in EKPHRASIS_TAG_MAPPING and EKPHRASIS_TAG_MAPPING[tag[1:-1]] # Only if a mapping exists
}

emoticons_extended = {**ekphrasis_filtered, **custom_emoticons}

# ----- Emoticon Processor Setup -----
emoticon_processor = TextPreProcessor(
    normalize=[],
    annotate={"emoticon"},
    fix_html=False,
    unpack_hashtags=False,
    unpack_contractions=False,
    spell_correct_elong=False,
    tokenizer=lambda s: s.split(),
    dicts=[emoticons_extended],
    lowercase=False
)


# ----- Emoji Conversion Setup -----
def emoji_processor(text):
    return emoji.replace_emoji(
        text,
        # Replace emoji with a matching tag with similar format to emoticons <tag>
        replace=lambda e, m: f"<{emoji.demojize(e).strip(':').replace('_', ' ').replace('-', ' ')}>"
    )


# Date and Time format preservation
date_and_time = [
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",          # e.g. 10/10 or 10/10/2025
    r"\b\d{1,2}-\d{1,2}(?:-\d{2,4})?\b",          # e.g. 10-10 or 10-10-2025
    r"\b\d{1,2}.\d{1,2}(?:.\d{2,4})?\b",          # e.g. 10.10 or 10.10.2025
    r"\b\d{1,2}:\d{2}(?:\s?[AaPp][Mm])?\b",       # e.g. 12:30 or 12:30 PM
    r"\b\d{1,2}[AaPp][Mm]\b",                     # e.g. 3am, 11PM
    r"\b\d{1,2}\s?[AaPp][Mm]\b",                  # e.g. 2 am, 3 Am
    r"\b(midnight|noon|morning|evening|afternoon|dawn|dusk|twilight)\b"
]


# Price format preservation
price = [
    r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",             # $1,000.00 or $100
    r"\$\s?\d+(?:\.\d{1,2})?",                            # $100.00 or $1.1
    r"£\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",              # £1,000.00
    r"£\s?\d+(?:\.\d{1,2})?",                             # £100.00
    r"€\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",              # €1,000.00
    r"€\s?\d+(?:\.\d{1,2})?",                             # €100.00
    r"\d+(?:\.\d{2})?\s?(dollars|pounds|euros)",          # 10 dollars, 10.00 pounds
    r"\d+(?:\.\d+)?\s?(k|m|million|billion)",             # 1k, 1.1M, 10 million
    r"(USD|GBP|EUR)\s?\d+(?:\.\d{1,2})?",                 # USD 100, GBP 100.00
    r"\d+(?:\.\d{1,2})?\s?(USD|GBP|EUR)"                  # 100.00USD
]


# ----- Normalise Text -----
def normaliser(text):
    text = str(text)
    text = contractions.fix(text) # Fix contractions dont → don't
    # To make sure important contractions are preserved
    text = re.sub(r"\b(can|could|did|would|should|is|are|was|were)(nt)\b", r"\1n't", text)
    # Pre-step: Remove Reddit tags like -r/movies or /r/movies
    text = re.sub(r'(?:[\-\/])?r\/[^\s]+', '', text)
    # Clean repetition of [] {} () / \ |
    text = re.sub(r'(?<=[\(\)\{\}\\\/\|\[\]])(?=[\(\)\{\}\\\/\|\[\]])', ' ', text)

    text = re.sub(r"\[NAME\]", "person", text, flags=re.IGNORECASE) # <name> → person
    text = re.sub(r"\[RELIGION\]", "religion", text, flags=re.IGNORECASE)
    text = re.sub(r"\[TEAM\]", "team", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(T|ALL)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r'https?:\/\/\S+', 'link', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', 'email', text, flags=re.IGNORECASE)
    text = re.sub(r'(@\s*)(me|you|him|her|us|them)', r'tag \2', text, flags=re.IGNORECASE) # @ pronoun → tag pronoun
    text = re.sub(r'([a-zA-Z]+)(@)([a-zA-Z]+)', r'\1\3', text, flags=re.IGNORECASE) # @ → a r'\1a\3' / drop
    text = re.sub(r'\?\s*@', '?', text) # Uncomplete username removal
    text = re.sub(r'@\s*\?', '?', text) # Uncomplete username removal
    text = re.sub(r'@{2,}', ' ', text, flags=re.IGNORECASE) # Repeated @
    # text = re.sub(r'\s*@\s*(\w+l+y)', r' \1', text, flags=re.IGNORECASE) # Remove @ before adverb/adjective
    # # text = re.sub(r'\s+@\s+', 'at', text) # Lonely @ → at
    text = re.sub(r"@\w+", "person", text, flags=re.IGNORECASE) # <user> → person
    text = re.sub(r'\s+@\s+', ' ', text) # Remove the rest of @
    text = re.sub(r"\d{5,}", '', text)  # Long numbers

    text = emoji_processor(text)
    text = " ".join(emoticon_processor.pre_process_doc(text))

    # Preserve emoji and emoticon tags e.g. <joy>, <thumbs up>, etc.
    preserved_tags = re.findall(r'<[^<>]+>', text)
    tag_map = {f"__TAG{index}__": tag for index, tag in enumerate(preserved_tags)}
    # Replace tags with placeholders
    for placeholder, tag in tag_map.items():
        text = text.replace(tag, placeholder)

    # Preserve date and time strings
    preserved_data_and_time = {}
    for i, pattern in enumerate(date_and_time):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        # Replace times and dates with placeholders
        for match in matches:
            key = f"__PRESERVED__{i}__{len(preserved_data_and_time)}__"
            preserved_data_and_time[key] = match
            text = text.replace(match, key)

    # Preserve price strings
    preserved_prices = {}
    for i, pattern in enumerate(price):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        # Replace prices with placeholders
        for match in matches:
            key = f"__PRICE__{i}_{len(preserved_prices)}__"
            preserved_prices[key] = match
            text = text.replace(match, key)

    # Remove unwanted characters
    # Keep letters, digits, space, and selected punctuation (., ,, !, ?, ', ")
    text = re.sub(r'[^\w\s\,\.\!\?\'\"]+', '', text)
    
    # Restore protected tags
    for placeholder, tag in tag_map.items():
        text = text.replace(placeholder, tag)

    # Restore date and time strings
    for placeholder, item in preserved_data_and_time.items():
        text = text.replace(placeholder, item)

    # Restore price strings
    for placeholder, item in preserved_prices.items():
        text = text.replace(placeholder, item)
    
    # Cleanup --normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove _ leftovers
    text = re.sub(r'_', '', text)

    # Remove angle brackets from emoji/emoticon tags e.g., <joy> → joy <girl with happy face> → girl with happy face
    text = re.sub(r"<([^<>]+)>", r"\1", text)

    return text.strip()


# ----- Load and Process -----
def load_and_prepare_split(name):
    df = pd.read_csv(INPUT_PATH / f"goemotions_{name}.csv")
    df["fine"] = df["fine"].apply(lambda x: [label.strip() for label in str(x).split(",") if label.strip()])
    df["text"] = df["text"].apply(normaliser)
    return df


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


def contain_rare(row_labels, label_counts, threshold=1000):
    """Check if a record contains any rare label"""
    return any(label_counts[label] < threshold for label in row_labels)


def filtering(row, label_counts, median):
    """Filter data"""
    # Keep if it has a valid length
    if len(row["text"].split()) >= MIN_WORDS:
        return True
    # Otherwise, only keep if it contains rare label
    else:
        return contain_rare(row["fine"], label_counts, threshold=median)


def load_split(split):
    """Loads and preprocesses data splits"""
    
    dataframe = pd.read_csv(INPUT_PATH / f"goemotions_{split}.csv")
    # Normalise comments (text)
    dataframe["text"] = dataframe["text"].apply(normaliser)
    # Convert string columns to lists and parse
    for column in ["fine", "intermediate", "coarse"]:
        dataframe[column] = dataframe[column].apply(parse_labels)
    
    return dataframe

# ----- Load Data -----
train = pd.read_csv(INPUT_PATH / "goemotions_train.csv")
dev = pd.read_csv(INPUT_PATH / "goemotions_dev.csv")
test = pd.read_csv(INPUT_PATH / "goemotions_test.csv")

train_dataframe = train.copy()
dev_dataframe = dev.copy()
test_dataframe = test.copy()

# Covert label columns to list format + parse + normalise text 
for dataframe in [train_dataframe, dev_dataframe, test_dataframe]:
    for column in ["fine", "intermediate", "coarse"]:
        dataframe[column] = dataframe[column].apply(parse_labels)
    dataframe["text"] = dataframe["text"].apply(normaliser)

# 1. Create a dataframe consisting of neutral-only labeled comments
neutral_pool = train_dataframe[train_dataframe["fine"].apply(lambda labels: labels == ["neutral"])]

# 2. Now Remove neutral records from train dataframe
train_dataframe = train_dataframe.drop(index=neutral_pool.index)
print(f"\nFound {len(neutral_pool)} neutral-only records in train data.")

# ///// Train data is neutral-only free and we have a neutral pool \\\\\

# First Tact: Remove 'neutral' from any multi-label record
train_dataframe["fine"] = train_dataframe["fine"].apply(
    lambda labels: [label for label in labels if label != "neutral"]
)
train_dataframe["intermediate"] = train_dataframe["intermediate"].apply(
    lambda labels: [label for label in labels if label != "neutral"]
)
train_dataframe["coarse"] = train_dataframe["coarse"].apply(
    lambda labels: [label for label in labels if label != "neutral"]
)

# Drop samples that became empty after removing 'neutral'
train_dataframe = train_dataframe[train_dataframe["fine"].apply(lambda x: len(x) > 0)].copy()

# Compute median
label_counts = Counter(label for labels in train_dataframe["fine"] for label in labels)
median = int(statistics.median(label_counts.values()))
print(f"\nMedian: {median}")

# Filter Train data
train_dataframe_filtered = train_dataframe[train_dataframe.apply(lambda row: filtering(row, label_counts, median), axis=1)]
print(f"\n{len(train_dataframe)} neutral free train samples.")

# Filter neutral-pool of short sentences for upsampling
neutral_pool = neutral_pool[neutral_pool["text"].apply(lambda x: isinstance(x, str) and len(x.split()) >= MIN_WORDS)]
print(f"\nFound {len(neutral_pool)} valid neutral-only samples.")

# Raise error if there isn't enough 'neutral' records ...
if len(neutral_pool) < median:
    raise ValueError(f"Not enough neutral-only samples to upsample: required={median}, available={len(neutral_pool)}")

# Sample 'neutral'-only records from the pool to match the median count of other emotion labels
sampled_neutral = neutral_pool.sample(n=median, random_state=42)
# Remove sampled records from the pool
neutral_pool = neutral_pool.drop(sampled_neutral.index)
# Add source metadata to the pool
neutral_pool["split"] = "neutral_pool"

# Append sampled neutrals to train set
train_selected = pd.concat([train_dataframe, sampled_neutral], ignore_index=True)

# ///// Train Data And Neutral Pool Are Ready -- Selection Complete \\\\\

# Convert back to comma separeted strings ...
for column in ["fine", "intermediate", "coarse"]:
    train_selected[column] = train_selected[column].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    dev_dataframe[column] = dev_dataframe[column].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    test_dataframe[column] = test_dataframe[column].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    neutral_pool[column] = neutral_pool[column].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

# Store ...
train_selected.to_csv(OUTPUT_PATH / "goemotions_train_selected.csv", index=False)
dev_dataframe.to_csv(OUTPUT_PATH / "goemotions_dev_selected.csv", index=False)
test_dataframe.to_csv(OUTPUT_PATH / "goemotions_test_selected.csv", index=False)
neutral_pool.to_csv(OUTPUT_PATH / "neutral_pool.csv", index=False)

print("\nSelected train, dev, test, and neutral pool datasets saved to:", OUTPUT_PATH)

summary = {
    "original_train_size": len(pd.read_csv(INPUT_PATH / "goemotions_train.csv")),
    "final_train_size": len(train_selected),
    "sampled_neutral": len(sampled_neutral),
    "pool_size": len(neutral_pool),
    "label_median": median,
    "label_distribution": Counter(label for row in train_selected['fine'] for label in row.split(', '))
}

with open(OUTPUT_PATH / "selection_summary.json", "w") as file:
    json.dump(summary, file, indent=2)

print(f"\nSummary information saved at {OUTPUT_PATH}\n")
