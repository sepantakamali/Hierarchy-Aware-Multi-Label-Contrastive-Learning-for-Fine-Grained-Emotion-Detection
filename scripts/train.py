from pathlib import Path
import json
import torch
import torch._dynamo
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
import random
from torch.amp import autocast, GradScaler
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from multilabel_contrastive_dataset import MultiLabelContrastiveDataset
from encoder import ContrastiveEncoder
from losses import PrototypicalContrastiveLoss
import torch.nn.functional as F
from prototypes import initialize_prototypes, update_prototypes
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from math import ceil
from sklearn.metrics import classification_report
import numpy as np
import os
import torch.nn.utils
import pandas as pd
from sklearn.metrics import silhouette_score
from collections import defaultdict
import numpy as np
import json
import pandas as pd
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score as _sk_silhouette_score
from sklearn.metrics import silhouette_samples as _sk_silhouette_samples


# ///////////////////// Attention ////////////////////////////////////////////////////////////////////
# The code is optimised for an NVIDIA A100 GPU with mixed precision training (TF32/FP16).
# It can run on other GPUs or CPU in standard FP32 mode, but may require
# disabling mixed precision or adjusting datatypes for compatibility.
# /////////////////////////////////////////////////////////////////////////////////////////////////////

# ///////////////////// Configurations ///////////////////////////////////////////////////////////////
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set default random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except AttributeError:
    pass

# Performance settings for Colab
torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable TF32 precision on A100 for faster matmuls and convolutions
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Debug mode flag
DEBUG_MODE = True

# ////////////////////////////////////////////////////////////////////////////////////////////////////


# Path setup
BASE_DIRECTORY = Path(__file__).resolve().parent.parent

# Load hierarchy dictionary
with open(BASE_DIRECTORY / "data" / "hierarchy_dictionary.json", "r", encoding="utf8") as file:
    hierarchy_dictionary = json.load(file)

# Set checkpoint directory
CHECKPOINT_DIRECTORY = BASE_DIRECTORY / "checkpoints"
CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEBUG_MODE:
    print(f"\nDevice: {device}\n")

embedder_name = "joeddav/distilbert-base-uncased-go-emotions-student"
# "microsoft/deberta-v3-large"

# ///////////////////// Batching Hyperparameters /////////////////////////////////////////////////////
label_frequency = 21
emotion_count = 27
emotion_share = label_frequency * emotion_count
# neutral_injection_ratio = 0.05  # Set to 0.0 to disable neutral injection
# neutral_share = max(label_frequency, ceil(neutral_injection_ratio * emotion_share) + 1)
neutral_share = 21 # Lets try equal shares
# Set batch_size to emotion_share + neutral_share
batch_size = emotion_share + neutral_share
# ////////////////////////////////////////////////////////////////////////////////////////////////////

lr = 1e-4  # learning rate

# Number of epochs
num_epochs = 15

# Smoothe prototype updates
momentum = 0.2

contrastive_learning_ne = 15  # Pure contrastive learning phase

# Epochs at which to record and save training embeddings
# record_epochs = {1, 8, 15, 21, 25}
record_epochs = {1, 8, 12, 15}
overfit_epochs = {6, 8, 10, 12} # Check for early stop

# Temporary storage for embeddings and labels
saved_embeddings, saved_labels = [], []

# Load tokeniser and the model
tokenizer_name = embedder_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
embedder = AutoModel.from_pretrained(embedder_name)
model = ContrastiveEncoder(encoder=embedder, projection_dim=256).to(device)
# Compile model for speed (requires PyTorch 2.0+)
model = torch.compile(model)
# Use channel-last memory format for improved GPU throughput
model = model.to(memory_format=torch.channels_last)

# Contrastive Learning Loss Function
loss_function = PrototypicalContrastiveLoss(
    hierarchy_dictionary=hierarchy_dictionary,
    embedding_dim=model.output_dim,
).to(device)

# Suppress Dynamo errors to fall back to eager when needed.
# Something must be wrong (probably dimension mismatch between embeddings and prototypes)
# Otherwise no need
# torch._dynamo.config.suppress_errors = True

# Optimise loss function
loss_function = torch.compile(loss_function)

# Ensure all parameters in the encoder are trainable
for param in model.encoder.parameters():
    param.requires_grad = True

# ///////////////////// Load Datasets ////////////////////////////////////////////////////////////////
# Always use the same label_list with "neutral" last for all datasets.
train_dataset = MultiLabelContrastiveDataset(
    csv_path=BASE_DIRECTORY / "data" / "preprocessed" / "goemotions_train_isolated.csv",
    tokenizer_name=tokenizer_name,
    label_list=None,  # Let dataset infer label list
    split="train",
)

# Retrieve and ensure "neutral" label is included as the last label
label_list = train_dataset.label_list
# Otherwise correct the structure
if label_list[-1] != "neutral":
    if DEBUG_MODE:
        print("[AMENDMENT] fixing neutral placement in labels list...")
        print(f"[AMENDMENT] original labels list: {label_list}")
    label_list = [label for label in label_list if label != "neutral"] + ["neutral"]
    train_dataset.label_list = label_list
    if DEBUG_MODE:
        print(f"[AMENDMENT] amended labels list: {label_list}")

if DEBUG_MODE:
    print(f"\n[STATE] labels list: {label_list}\n")

# Use full label list (with "neutral" last) for both validation and neutral datasets
validation_dataset = MultiLabelContrastiveDataset(
    csv_path=BASE_DIRECTORY / "data" / "preprocessed" / "goemotions_dev_selected.csv",
    tokenizer_name=tokenizer_name,
    label_list=label_list,
    split="validation",
)

neutral_dataset = MultiLabelContrastiveDataset(
    csv_path=BASE_DIRECTORY / "data" / "preprocessed" / "neutral_isolated.csv",
    tokenizer_name=tokenizer_name,
    label_list=label_list,
    neutral_only=True,  # This flag ensures only the last label index is active
    split="neutral",
)
# ////////////////////////////////////////////////////////////////////////////////////////////////////

def build_emotion_embeddings_mappings(dataset):
    """Map each emotion label to the list of samples' indices containing it."""
    label_to_indices = defaultdict(list)
    for index in range(len(dataset)):
        labels = dataset[index]["labels"]
        active = (labels == 1).nonzero(as_tuple=True)[0].tolist()
        for label_index in active:
            label_to_indices[label_index].append(index)
    return label_to_indices

def build_neutral_indices(dataset, neutral_indices):
    """Return all indices in the dataset that are labeled as neutral."""
    indices = []
    for index in range(len(dataset)):
        # Double check to make sure only neutral instances are passed as neutral indices
        labels = dataset[index]["labels"]
        if labels[neutral_indices] == 1:
            indices.append(index)
    return indices

# Custom collate function for prototype initialization
def custom_collate(batch):
    output = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            stacked = torch.stack([item[key] for item in batch])
            output[key] = stacked
        elif isinstance(batch[0][key], list):
            output[key] = [item[key] for item in batch]
        else:
            output[key] = [item[key] for item in batch]
    return output

# ///////////////////// Prototypes ///////////////////////////////////////////////////////////////////
# Creating a combined emotion + neutral dataset to initiate prototypes
indices = list(range(len(neutral_dataset)))
sampled_indices = random.sample(
    population=indices,
    k=77 # Min frequency
    )
neutral_sampled = Subset(neutral_dataset, sampled_indices)
prototypes_initiator_set = ConcatDataset([train_dataset, neutral_sampled])

# Try retrieving prototypes embeddings
prototype_cache_path = CHECKPOINT_DIRECTORY / "initial_prototypes.pt"

if prototype_cache_path.exists(): # Prototypes exist
    if DEBUG_MODE:
        print(f"\n[Prototype Cache] Loading tensor cache from {prototype_cache_path}\n")
    prototypes = torch.load(prototype_cache_path, map_location=device).to(device)
else: # Prototypes does not exist
    if DEBUG_MODE:
        print("\n[Prototype Cache] No cache found. Initializing prototypes tensor...\n")

    # Load data embeddings from prepared set    
    combined_loader = DataLoader(
        prototypes_initiator_set,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True
    )

    # Initialize prototypes using the selected data
    # initialize_prototypes returns a Tensor of shape [C, D]
    prototypes = initialize_prototypes(
        dataloader=combined_loader,
        encoder=model,
        label_list=label_list,
        device=device
    )

    # Cache prototypes
    torch.save(prototypes.cpu(), prototype_cache_path)
    # Send prototypes to device for training
    prototypes = prototypes.to(device)
# ////////////////////////////////////////////////////////////////////////////////////////////////////

optimizer = None
# scheduler = None
scaler = GradScaler()

# # Early stopping parameters
# patience = 5 # to have at least 15 -> 5 -> 5 epochs for each phase
# best_validation_loss = float("inf")
# epochs_without_improvement = 0

# ///////////////////// Prepare for Training /////////////////////////////////////////////////////////

# ///////////////////// Variables needed for custom batching /////////////////////////////////////////

# Build a mapping for each emotion label to its instance embeddings in train data
emotions_to_indices = build_emotion_embeddings_mappings(train_dataset)
# Track neutral indices to add 
neutral_indices = build_neutral_indices(neutral_dataset, label_list.index("neutral"))
# Neutral indices with offset to be put after emotion data
neutral_indices_offsetted = [len(train_dataset) + index for index in neutral_indices]

class LabelAwareBatchSampler(Sampler):
    def __init__(self, emotions_to_indices, neutral_indices_offsetted, label_count, label_frequency, neutral_share, shuffle_each_epoch=True):
        self.emotions_to_indices = {label: indices[:] for label, indices in emotions_to_indices.items()}
        self.neutral_indices_offsetted = neutral_indices_offsetted[:]
        self.label_count = label_count
        self.label_frequency = label_frequency
        self.emotion_share = label_count * label_frequency
        self.neutral_share = neutral_share
        self.shuffle = shuffle_each_epoch
        # Flat list of indices used in the current epoch
        self.current_epoch_indices = []

        # self.labels = []

    def __iter__(self):
        # Reset the record of indices each epoch
        self.current_epoch_indices = []

        if self.shuffle:
            # Shuffle emotions indices
            for indices in self.emotions_to_indices.values():
                random.shuffle(indices)
            # Shuffle neutral indices
            random.shuffle(self.neutral_indices_offsetted)

        cursors = {label: 0 for label in range(self.label_count)}
        neutral_cursor = 0
        max_label_instances = max(len(indices) for indices in self.emotions_to_indices.values())
        # Calculate number of batches based on the maximum count of instances for labels
        num_batches = ceil(max_label_instances * self.label_count / self.emotion_share)

        # Pack the batches...
        for each_batch in range(num_batches):
            batch = []
            # labels = []
            # Track indices already used in this batch to ensure uniqueness
            used_indices = set()
            # Emotion instance picking...
            for label in range(self.label_count):
                indices = self.emotions_to_indices[label]
                for each_label in range(self.label_frequency):
                    # Reuse instances for rare emotions --issue
                    while cursors[label] < len(indices):
                        candidate_index = indices[cursors[label]]
                        if candidate_index in used_indices:
                            cursors[label] += 1
                            continue
                        batch.append(candidate_index)
                        # labels.append(label)
                        used_indices.add(candidate_index)
                        cursors[label] += 1
                        break
                    else:
                        # If we've exhausted all indices, reset and reshuffle if needed
                        if self.shuffle:
                            random.shuffle(indices)
                        cursors[label] = 0
                        # After resetting, pick the first unique index
                        candidate_index = indices[cursors[label]]
                        # Skip already used samples to avoid repetition
                        if candidate_index in used_indices:
                            cursors[label] += 1
                            continue
                        batch.append(candidate_index)
                        # labels.append(label)
                        used_indices.add(candidate_index)
                        cursors[label] += 1

            # Neutral injection...
            for each_share in range(self.neutral_share):
                # Same reset strategy...
                while neutral_cursor < len(self.neutral_indices_offsetted):
                    candidate_index = self.neutral_indices_offsetted[neutral_cursor]
                    if candidate_index in used_indices:
                        neutral_cursor += 1
                        continue
                    batch.append(candidate_index)
                    # labels.append(label)
                    used_indices.add(candidate_index)
                    neutral_cursor += 1
                    break
                else:
                    # If we've exhausted all neutral indices, reset and reshuffle if needed
                    if self.shuffle:
                        random.shuffle(self.neutral_indices_offsetted)
                    neutral_cursor = 0
                    candidate_index = self.neutral_indices_offsetted[neutral_cursor]
                    # Skip already used samples to avoid repetition
                    if candidate_index in used_indices:
                        neutral_cursor += 1
                        continue
                    batch.append(candidate_index)
                    # labels.append(label)
                    used_indices.add(candidate_index)
                    neutral_cursor += 1

            # Batching complete...

            # Shuffle the batch 
            if self.shuffle:
                random.shuffle(batch)

            # Record these sample indices in flattened order
            self.current_epoch_indices.extend(batch)

            # Pass the batch to sampler
            yield batch

    def __len__(self):
        max_label_instances = max(len(indices) for indices in self.emotions_to_indices.values())
        return ceil(max_label_instances * self.label_count / self.emotion_share)

# ///////////////////// Data Loaders /////////////////////////////////////////////////////////////////
train_batch_sampler = LabelAwareBatchSampler(
    emotions_to_indices=emotions_to_indices,
    neutral_indices_offsetted=neutral_indices_offsetted,
    label_count=emotion_count,
    label_frequency=label_frequency,
    neutral_share=neutral_share,
    shuffle_each_epoch=True
)

# Combine train and neutral datasets for a single DataLoader
combined_dataset = ConcatDataset([train_dataset, neutral_dataset])

# Train sampler with custom batcher to controll neutral injection and balanced training
train_loader = DataLoader(
    combined_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=custom_collate,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
# Validation dataset loader
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate,
    num_workers=4,
    pin_memory=True
)
# ///////////////////// Validation related functions //////////////////////////////////////////////////////////////////////////

def build_balanced_indices_for_validation(dataset, per_label=200):
    """Balanced subset indices: up to per_label samples per class."""
    C = len(dataset.label_list) # C classes
    label_to_index = [[] for each_class in range(C)]
    for i in range(len(dataset)):
        label_vector = dataset[i]["labels"] # Label vector of sample i
        active_classes = (label_vector == 1).nonzero(as_tuple=True)[0].tolist()
        for c in active_classes:
            label_to_index[c].append(i) # Append sample index i to the label list
        # # Now label_to_index[c] is the list of sample indices that belong to class c
    
    rng = np.random.default_rng(123) # Random generator with fixed seed for reproducibility
    keep = []
    for c in range(C):
        indices = label_to_index[c]
        if not indices: # No sample for these class? skip.
            continue
        if len(indices) > per_label: # Cap the sample size of classes to per_label
            indices = rng.choice(indices, size=per_label, replace=False)
        keep.append(np.array(indices, dtype=int)) # Append the sampled samples to the labels list
    # Return the balanced, unique, validation samples for all classes
    return np.unique(np.concatenate(keep)).tolist()

def collect_validation_scores(model, prototypes, loader, device):
    """Return labels Y and scores S (normalised [0 to 1]) for all validation samples."""
    model.eval()
    V, Y = [], []
    with torch.inference_mode(): # Faster and safer than torch.eval()
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16): # Run under mixed precision --speeds up
                e = model(input_ids=ids, attention_mask=mask) # Batch embeddings
            e = F.normalize(e, dim=1)
            # Store embeddings and labels of the batch (on CPU)!
            V.append(e.cpu()) # [N, D] samples x embedding dimension
            Y.append(labels.cpu())
    # Concatenate embeddings and labels of all batches (on CPU)!
    V = torch.cat(V, dim=0)
    Y = torch.cat(Y, dim=0).int()
    P = F.normalize(prototypes, dim=1).detach().cpu() # Normalised prototypes (on CPU)!
    # Embeddings and prototypes now have unit length --ready for cosine similarity
    s = (V @ P.T).clamp(-1, 1)
    S = (s + 1) / 2.0  # Rescale to [0, 1]
    # Return true labels and prototype similarity scores for each sample
    return Y.numpy(), S.numpy()

def tune_thresholds(S, Y, label_names):
    """Thresholds tuner to find general threshold of each label in multi-label context."""
    coarse = np.linspace(0.05, 0.95, 19) # Coarse level thresholds 0.05 steps
    C = Y.shape[1]
    thresholds, per_class_f1 = {}, {}
    for c in range(C):
        y_true, s = Y[:, c], S[:, c] # y_true: gold labels of class c. s: similarity scores of samples to prototype c
        # Default threshold: 0.5
        # Default f1 score: 0.0
        if y_true.sum() == 0:
            thresholds[label_names[c]] = 0.5
            per_class_f1[label_names[c]] = 0.0
            continue
        # Find coarse threshold
        best_t, best_f1 = 0.5, 0.0
        for t in coarse:
            # Calculate F1 based on coarse threshold t
            f1 = f1_score(y_true, (s >= t).astype(int), zero_division=0)
            # Update if better F1 found
            if f1 > best_f1:
                best_f1, best_t = f1, t
        # Find fine thresholds --0.01 steps in 0.05 neighbourhood of the chosen coarse threshold
        fine = np.clip(np.linspace(best_t-0.05, best_t+0.05, 11), 0.01, 0.99)
        for t in fine:
            # Calculate F1 based on fine threshold t
            f1 = f1_score(y_true, (s >= t).astype(int), zero_division=0)
            # Update if better F1 found
            if f1 > best_f1:
                best_f1, best_t = f1, t
        # Store thresholds and per-class F1s
        thresholds[label_names[c]] = float(best_t)
        per_class_f1[label_names[c]] = float(best_f1)
    
    T = np.array([thresholds[label] for label in label_names])[None, :] # All thresholds
    Y_pred = (S >= T).astype(int) # Final predictions with tuned thresholds
    
    metrics = {
        "f1_micro": f1_score(Y, Y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(Y, Y_pred, average="macro", zero_division=0),
        "f1_samples": f1_score(Y, Y_pred, average="samples", zero_division=0),
        "precision_micro": precision_score(Y, Y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(Y, Y_pred, average="micro", zero_division=0),
    }

    return thresholds, per_class_f1, metrics

# ////////////////////////////////////////////////////////////////////////////////////////////////////

# ///////////////////// Configurations ///////////////////////////////////////////////////////////////
# current_thresholds = {label: 0.5 for label in label_list}  # Default thresholds
# best_thresholds_phase1 = {label: 0.5 for label in label_list}
# best_f1s_phase1 = {label: 0.0 for label in label_list}
# reinit_scheduler_phase2 = False
# momentum_schedule = [0.80 + (e / 14) * (0.95 - 0.80) for e in range(contrastive_learning_ne)]
# ////////////////////////////////////////////////////////////////////////////////////////////////////

# ///////////////////// Train Body ///////////////////////////////////////////////////////////////////
for epoch in range(num_epochs):

    # Clear embeddings and labels lists
    saved_embeddings.clear()
    saved_labels.clear()

    # Announce current phase
    if epoch < contrastive_learning_ne:
        phase_name = "Contrastive Learning"
        if epoch < 4:
            momentum = 0.5
        elif epoch < 6:
            momentum = 0.45
        elif epoch < 8:
            momentum = 0.4
        else:
            momentum = 0.3
    
    print(f"\n>>>Starting Epoch {epoch+1}/{num_epochs} Phase: {phase_name}")
    
    # initialize per-phase loss trackers
    contrastive_learning_total_loss = 0.0

    for parameter in model.parameters():
        parameter.requires_grad = True
    prototypes_update = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = None

    model.train()
    # Accumulators for per-epoch prototype recentring
    epoch_prototypes_sum = torch.zeros_like(prototypes)                 # [C, D]
    epoch_prototypes_count = torch.zeros(prototypes.size(0), device=device)  # [C]
    total_loss = 0
    for batch_index, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)  

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        # Mixed precision forward
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = F.normalize(embeddings, dim=1)
            # Record embeddings and labels for selected epochs
            if epoch+1 in record_epochs:
                saved_embeddings.append(embeddings.detach().cpu())
                saved_labels.append(labels.detach().cpu())
            
            # Accumulate per-label sums for prototype recenter (multi-label aware)
            with torch.no_grad():
                # labels: [B, C] (0/1); embeddings: [B, D] (L2-normalized)
                # Sum embeddings per class via label^T @ e → [C, D]
                epoch_prototypes_sum += (labels.float().T @ embeddings.float())
                # Count of positives per class in this batch
                epoch_prototypes_count += labels.float().sum(dim=0)
        
            contrastive_learning_loss = loss_function(
                embeddings,
                labels,
                prototypes,
                epoch
            )
        
        # Track per-phase losses
        contrastive_learning_total_loss += contrastive_learning_loss.item()
        
        # Total loss
        total_loss_combined = contrastive_learning_loss

        # Backward and optimize
        scaler.scale(total_loss_combined).backward()
        # unscale then clip gradients to avoid unstable BCE spikes
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Soft prototype updates every step
        if prototypes_update:
            with torch.no_grad():
                prototypes = update_prototypes(
                    prototypes,
                    embeddings,
                    labels,
                    momentum=momentum
                )

        # Collect total loss
        total_loss += total_loss_combined.item()
    
    # Gentle prototype recenter at the end of selected epochs
    # Apply after epochs 4, 8, 12
    if epoch in (3, 7, 11):
        with torch.no_grad():
            # Compute per-class means where we have at least one positive
            mask = epoch_prototypes_count > 0
            if mask.any():
                means = torch.zeros_like(prototypes)
                means[mask] = F.normalize(
                    epoch_prototypes_sum[mask] / epoch_prototypes_count[mask].unsqueeze(1), dim=1
                )
                # Blend 10% of the fresh means into current prototypes, then renormalize
                prototypes[mask] = F.normalize(0.9 * prototypes[mask] + 0.1 * means[mask], dim=1)
                # Alert
                if DEBUG_MODE:
                    seen = int(mask.sum().item())
                    print(f"[Recenter] Blended recenter applied to {seen} classes at epoch {epoch+1} (10% blend)")
            else: # Just to be safe
                if DEBUG_MODE:
                    print(f"[Recenter] No positives seen this epoch; skipping recenter at epoch {epoch+1}")
    # /////////////////////////// Metrics ///////////////////////////

    # Save epoch embeddings, labels, prototypes for visualization and cluster metrics only for selected epochs
    if epoch+1 in record_epochs:
        # Always save embeddings and prototypes for record_epochs
        # Ensure normalized copies for metrics (safety)
        epoch_embeddings = torch.cat(saved_embeddings, dim=0).to(device)
        epoch_labels = torch.cat(saved_labels, dim=0)
        E = F.normalize(epoch_embeddings, p=2, dim=1)
        P = F.normalize(prototypes, p=2, dim=1)
        torch.save(E, CHECKPOINT_DIRECTORY / f"train_embeddings_epoch{epoch+1}.pt")
        torch.save(epoch_labels, CHECKPOINT_DIRECTORY / f"train_labels_epoch{epoch+1}.pt")
        torch.save(P.cpu().clone(), CHECKPOINT_DIRECTORY / f"prototypes_epoch{epoch+1}.pt")

        # Per-label centroid distances --multi-label
        per_label_distances = {}
        total_distances_sum = 0.0
        total_count = 0
        for c in range(len(label_list)):
            mask = epoch_labels[:, c].bool()
            if mask.any():
                # cosine similarity between samples of class c and prototype c
                s = (E[mask] * P[c]).sum(dim=1)
                distance = 1.0 - s  # cosine distance
                per_label_distances[label_list[c]] = distance.mean().item()
                total_distances_sum += distance.sum().item()
                total_count += int(mask.sum().item())
            else:
                per_label_distances[label_list[c]] = float('nan')
        
        # Weighted overall average distance across all samples labels
        average_to_prototype_distance = (total_distances_sum / max(total_count, 1))

        # Prototype separation
        p_p = P @ P.T # prototype to prototype similarity
        p_p.fill_diagonal_(1.0)
        distances = 1.0 - p_p                                    # cosine distance
        distances.fill_diagonal_(float('inf'))
        # closest pair (smallest distance)
        min_prototypes_distance = distances.min().item()
        # average off-diagonal distance
        off = ~torch.eye(len(label_list), dtype=torch.bool, device=distances.device)
        mean_offdiagonal_distance = distances[off].mean().item()

        # Save combined cluster metrics as CSV (one row)
        combined_metrics_dataframe = pd.DataFrame([{
            "Epoch": epoch + 1,
            "Mean offdiagonal prototype to prototype distance": mean_offdiagonal_distance,
            "Min prototypes distance": min_prototypes_distance,
            **{f"{label} average cosine distance": d for label, d in per_label_distances.items()},
        }])

        combined_metrics_dataframe.to_csv(
            CHECKPOINT_DIRECTORY / f"cluster_metrics_epoch{epoch+1}.csv",
            index=False
        )

        # Multi-label silhouette score based on shared-label
        X_np = E.cpu().numpy()                                 # L2-normalized embeddings
        Y_np = epoch_labels.cpu().numpy().astype(bool)         # multi-hot -> bool

        k = 32  # Fixed for compatibility --for tuning purposes only
        nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(X_np)
        distance, index = nn.kneighbors(X_np)
        # Drop self
        distance, index = distance[:, 1:], index[:, 1:]

        # For each sample i, a k × C matrix of neighbor labels
        Y_neighbors = Y_np[index] # [N, k, C]
        # share_any[i, j] = True if neighbor j shares ≥1 label with sample i
        share_any = np.any(Y_neighbors & Y_np[:, None, :], axis=2) # [N, k] Boolian!

        # Compute A (within) and B (between) means per sample; skip if a group is empty
        A = np.array([d[m].mean() if m.any() else np.nan for d, m in zip(distance, share_any)])
        B = np.array([d[~m].mean() if (~m).any() else np.nan for d, m in zip(distance, share_any)])

        valid = ~np.isnan(A) & ~np.isnan(B)
        s = (B - A) / np.maximum(A, B)
        multilabel_silhouette = float(np.nanmean(s[valid])) if valid.any() else float("nan")
        valid_fraction = float(valid.mean())
        mean_A = float(np.nanmean(A[valid])) if valid.any() else float("nan")
        mean_B = float(np.nanmean(B[valid])) if valid.any() else float("nan")

        # print(f"[Metrics] Multi-label silhouette (k={k}): {multilabel_silhouette:.4f} | valid={valid_fraction:.3f} | A={mean_A:.4f} | B={mean_B:.4f}")

        # Per-label silhouette scores --only at the last epoch to reduce run time (expensive)!
        # Uses the SAME k-NN neighbors (index, distance) computed above
        # Also computes a full (sklearn) silhouette within the subset of samples that carry label c.

        if epoch == 14:  # Epoch 15
            per_label_silhouette_knn = {}
            per_label_silhouette_full = {}
            for c, label_name in enumerate(label_list):
                # Anchors that have label c
                mask_c = Y_np[:, c]
                Nc = int(mask_c.sum())
                # We need at least two positives
                if Nc < 2:
                    per_label_silhouette_knn[label_name] = float("nan")
                    per_label_silhouette_full[label_name] = float("nan")
                    continue

                # k-NN restricted silhouette shared-label positives
                dc = distance[mask_c]
                index_c  = index[mask_c]
                neighbor_has_c = Y_np[index_c, c]

                A_vals = np.array([d[m].mean() if m.any() else np.nan for d, m in zip(dc, neighbor_has_c)])
                B_vals = np.array([d[~m].mean() if (~m).any() else np.nan for d, m in zip(dc, neighbor_has_c)])

                valid_sub = ~np.isnan(A_vals) & ~np.isnan(B_vals)
                if valid_sub.any():
                    s_sub = (B_vals - A_vals) / np.maximum(A_vals, B_vals)
                    per_label_silhouette_knn[label_name] = float(np.nanmean(s_sub[valid_sub]))
                else:
                    per_label_silhouette_knn[label_name] = float("nan")

                # Full sklearn silhouette (binary: label-c vs. others, on the FULL set) ===
                # Using the full dataset with binary labels ensures at least two clusters for silhouette.
                # We then average the per-sample silhouette values only over the positives of class c.
                # This one is more accurate but general, k-NN is more about per class saturation and noise detection
                # This silhouette should be around 0.5 while above silhouette around ±0.0 for multi-label overlap nature
                try:
                    y_active = (Y_np[:, c]).astype(int)
                    positives = int(y_active.sum()); neg = int(len(y_active) - y_active.sum())
                    if positives >= 2 and neg >= 2:
                        silhouette_all = _sk_silhouette_samples(X_np, y_active, metric="cosine")  # [N]
                        per_label_silhouette_full[label_name] = float(np.nanmean(silhouette_all[y_active == 1]))
                    else:
                        per_label_silhouette_full[label_name] = float("nan")
                except Exception:
                    per_label_silhouette_full[label_name] = float("nan")

        # Intra-cluster variance --per label
        intra_variances = []
        for c in range(len(label_list)):
            mask = epoch_labels[:, c].bool()
            if mask.any():
                intra_variances.append(E[mask].var(dim=0).mean().item())
        average_intra_variance = float(sum(intra_variances) / max(len(intra_variances), 1))

        # Inter-cluster variance --prototype spread
        inter_variance = float(P.var(dim=0).mean().item())

        # Separation index...
        separation_index = inter_variance / (average_intra_variance + 1e-9)

        print(
            f"[Metrics] Epoch {epoch+1} → Silhouette: {multilabel_silhouette:.4f} | "
            f"AverageToPrototypeDistance: {average_to_prototype_distance:.4f} | "
            f"IntraVarariance: {average_intra_variance:.4f} | InterVariance: {inter_variance:.4f} | "
            f"SeparationIndex: {separation_index:.4f}"
        )

        # --- Save extended clustering metrics as CSV ---
        row = {
            "epoch": epoch + 1,
            "multilabel_silhouette": float(multilabel_silhouette),
            "multilabel_silhouette_k32": float(multilabel_silhouette),
            "multilabel_silhouette_valid_fraction": float(valid_fraction),
            "multilabel_silhouette_mean_A": float(mean_A),
            "multilabel_silhouette_mean_B": float(mean_B),
            "average_to_prototype_distance": float(average_to_prototype_distance),
            "intra_variance": float(average_intra_variance),
            "inter_variance": float(inter_variance),
            "separation_index": float(separation_index),
        }

        # Merge per-label silhouettes only at the last epoch
        if epoch == 14 and 'per_label_silhouette_knn' in locals():
            for label_name in label_list:
                if label_name in per_label_silhouette_knn:
                    row[f"{label_name}_silhouette_knn"] = per_label_silhouette_knn[label_name]
                if 'per_label_silhouette_full' in locals() and label_name in per_label_silhouette_full:
                    row[f"{label_name}_silhouette_full"] = per_label_silhouette_full[label_name]

        # Add per-label average distances
        for label, v in per_label_distances.items():
            row[f"{label}_average_cosine_distance"] = float(v) if v == v else np.nan
        
        # KNN Classification --multi-label, per-label voting

        X = E.cpu().numpy()
        Y = epoch_labels.cpu().numpy().astype(int)

        k = 50
        nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(X)
        distance, index = nn.kneighbors(X)
        index = index[:, 1:] # Drop self

        predictions_knn = np.zeros_like(Y)
        half_k = k / 2.0
        for i in range(Y.shape[0]):
            neighbor_labels = Y[index[i]]
            # If more than half of the neighbors have label i --majority vote
            predictions_knn[i] = (neighbor_labels.sum(axis=0) > half_k).astype(int)

        exact_match_accuracy = (predictions_knn == Y).all(axis=1).mean()
        macro_f1 = f1_score(Y, predictions_knn, average='macro', zero_division=0)
        micro_f1 = f1_score(Y, predictions_knn, average='micro', zero_division=0)
        sample_f1 = f1_score(Y, predictions_knn, average='samples', zero_division=0)

        row["knn_exact_match_accuracy"] = float(exact_match_accuracy)
        row["knn_macro_f1"] = float(macro_f1)
        row["knn_micro_f1"] = float(micro_f1)
        row["knn_sample_f1"] = float(sample_f1)

        print(
            f"[Metrics] Epoch {epoch+1} → KNN — exact: {exact_match_accuracy:.4f} | "
            f"F1(micro/macro/samples): {micro_f1:.4f}/{macro_f1:.4f}/{sample_f1:.4f}"
        )

        # Save the extended clustering metrics row for this epoch
        dataframe = pd.DataFrame([row])
        dataframe.to_csv(CHECKPOINT_DIRECTORY / f"clustering_metrics_epoch{epoch+1}.csv", index=False)
        if DEBUG_MODE:
            print(f"[Metrics] Saved extended clustering metrics to {CHECKPOINT_DIRECTORY / f'clustering_metrics_epoch{epoch+1}.csv'}")
        

    num_batches = len(train_loader)

    print(f"Epoch {epoch+1} Contrastive Loss: {contrastive_learning_total_loss/num_batches:.4f}")

    average_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1}/{num_epochs} | Train Loss: {average_loss:.4f}\n")

    if epoch + 1 == 15: # Threshold tuning phase
        print(f">>> Contrastive phase ended at epoch {epoch+1} <<<\n")
        print("\n[ThresholdTuning] Running per-label threshold tuning...")
        # Balanced subset for fair calibration
        tune_idices = build_balanced_indices_for_validation(validation_dataset, per_label=200)
        validation_tune_subset = Subset(validation_dataset, tune_idices)
        validation_tune_loader = DataLoader(
            validation_tune_subset, batch_size=batch_size,
            collate_fn=custom_collate, num_workers=4, pin_memory=True
        )
        # Collect scores
        Yb, S = collect_validation_scores(model, prototypes, validation_tune_loader, device)
        # Tune thresholds
        thresholds, per_class_f1, tuned_metrics = tune_thresholds(S, Yb, label_list)
        # Save results
        thresholds_path = CHECKPOINT_DIRECTORY / f"thresholds_epoch{epoch+1}.json"
        with open(thresholds_path, "w") as f: json.dump(thresholds, f, indent=2)
        pd.DataFrame([tuned_metrics]).to_csv(CHECKPOINT_DIRECTORY / f"validation_thresholds_epoch{epoch+1}_summary.csv", index=False)
        pd.DataFrame([per_class_f1]).to_csv(CHECKPOINT_DIRECTORY / f"validation_thresholds_epoch{epoch+1}_per_class_f1.csv", index=False)
        print(f"[ThresholdTuning] Saved thresholds → {thresholds_path}")
        print(f"[ThresholdTuning] F1 micro/macro/samples: "
              f"{tuned_metrics['f1_micro']:.4f} / {tuned_metrics['f1_macro']:.4f} / {tuned_metrics['f1_samples']:.4f}")

        print(f">>> Thresold tuning phase ended at epoch {epoch+1}\n")
        break

# ////////////////////////////////////////////////////////////////////////////////////////////////////

# ///////////////// Full validation evaluation with tuned thresholds phase three /////////////////////////////////
print(">>> Starting inference phase\n")

thresholds_path = CHECKPOINT_DIRECTORY / "thresholds_epoch15.json"
with open(thresholds_path, "r") as f:
    tuned_thresholds = json.load(f)

# Ensure neutral is last in label_list
assert label_list[-1] == "neutral", "Expected 'neutral' to be the last label."

# Build threshold vector in label order; default to 0.5 if a label is missing in the JSON
thresholds_list = []
missing = []
for name in label_list:
    if name in tuned_thresholds:
        thresholds_list.append(float(tuned_thresholds[name]))
    else:
        thresholds_list.append(0.5)
        missing.append(name)
if missing:
    print(f"[WARNING] Using default 0.5 threshold for labels missing in JSON: {missing}")

T = np.array(thresholds_list, dtype=np.float32)[None, :]  # Thresholds [1, C]

# Full validation loader (imbalanced, raw distribution)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate,
    num_workers=4,
    pin_memory=True
)

# Collect embeddings and scores
Y_full_validation, S_full_validation = collect_validation_scores(model, prototypes, validation_loader, device)

# Predict with tuned thresholds
Y_pred_full = (S_full_validation >= T).astype(int)

# Compute sklearn metrics
full_val_metrics = {
    "exact_match": (Y_pred_full == Y_full_validation).all(axis=1).mean(),
    "f1_micro":  f1_score(Y_full_validation, Y_pred_full, average="micro",  zero_division=0),
    "f1_macro":  f1_score(Y_full_validation, Y_pred_full, average="macro",  zero_division=0),
    "f1_samples": f1_score(Y_full_validation, Y_pred_full, average="samples", zero_division=0),
    "precision_micro": precision_score(Y_full_validation, Y_pred_full, average="micro", zero_division=0),
    "recall_micro":  recall_score(Y_full_validation, Y_pred_full,  average="micro", zero_division=0),
}

print("\n[Full Validation] exact-match: "
      f"{full_val_metrics['exact_match']:.4f}")
print("[Full Validation] F1 micro/macro/samples: "
      f"{full_val_metrics['f1_micro']:.4f} / "
      f"{full_val_metrics['f1_macro']:.4f} / "
      f"{full_val_metrics['f1_samples']:.4f}")
print("[Full Validation] Precision / Recall (micro): "
      f"{full_val_metrics['precision_micro']:.4f} / "
      f"{full_val_metrics['recall_micro']:.4f}")

# Per-class F1s
per_class_f1 = f1_score(Y_full_validation, Y_pred_full, average=None, zero_division=0)
per_class_dataframe = pd.DataFrame([per_class_f1], columns=label_list)

# Save metrics + artifacts for later analysis
pd.DataFrame([full_val_metrics]).to_csv(
    CHECKPOINT_DIRECTORY / "validation_full_metrics_epoch15.csv", index=False
)
per_class_dataframe.to_csv(
    CHECKPOINT_DIRECTORY / "validation_full_per_class_f1_epoch15.csv", index=False
)
np.save(CHECKPOINT_DIRECTORY / "validation_full_y_true_epoch15.npy", Y_full_validation)
np.save(CHECKPOINT_DIRECTORY / "validation_full_scores_epoch15.npy", S_full_validation)
np.save(CHECKPOINT_DIRECTORY / "validation_full_y_pred_epoch15.npy", Y_pred_full)
with open(CHECKPOINT_DIRECTORY / "validation_full_thresholds_used_epoch15.json", "w") as f:
    json.dump({k: float(v) for k, v in zip(label_list, T.flatten().tolist())}, f, indent=2)

print(">>> Starting test phase\n")

# Test Set Evaluation
print("\n[TEST] Evaluating on test set with tuned thresholds from epoch 15...")

# Build test dataset using the SAME label_list
test_dataset = MultiLabelContrastiveDataset(
    csv_path=BASE_DIRECTORY / "data" / "preprocessed" / "goemotions_test_selected.csv",
    tokenizer_name=tokenizer_name,
    label_list=label_list,
    split="test",
)

assert label_list[-1] == "neutral", "Expected 'neutral' to be the last label."

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=custom_collate,
    num_workers=4,
    pin_memory=True
)

thresholds_path_base   = CHECKPOINT_DIRECTORY / "thresholds_epoch15.json"

with open(thresholds_path_base, "r") as f:
    tuned_thresholds = json.load(f)
print("[TEST] Using per-class thresholds:", thresholds_path_base.name)

# Build threshold vector in label order; default to 0.5 if a label is missing in the JSON
thresholds_list = []
missing = []
for name in label_list:
    if name in tuned_thresholds:
        thresholds_list.append(float(tuned_thresholds[name]))
    else:
        thresholds_list.append(0.5)
        missing.append(name)
if missing:
    print(f"[WARNING][TEST] Using default 0.5 threshold for labels missing in JSON: {missing}")
T_test = np.array(thresholds_list, dtype=np.float32)[None, :]

Y_test, S_test = collect_validation_scores(model, prototypes, test_loader, device)

Y_pred_test = (S_test >= T_test).astype(int)

test_metrics = {
    "exact_match": (Y_pred_test == Y_test).all(axis=1).mean(),
    "f1_micro":  f1_score(Y_test, Y_pred_test, average="micro",  zero_division=0),
    "f1_macro":  f1_score(Y_test, Y_pred_test, average="macro",  zero_division=0),
    "f1_samples": f1_score(Y_test, Y_pred_test, average="samples", zero_division=0),
    "precision_micro": precision_score(Y_test, Y_pred_test, average="micro", zero_division=0),
    "recall_micro":  recall_score(Y_test, Y_pred_test,  average="micro", zero_division=0),
}

print("\n[TEST] exact-match: "
      f"{test_metrics['exact_match']:.4f}")
print("[TEST] F1 micro/macro/samples: "
      f"{test_metrics['f1_micro']:.4f} / "
      f"{test_metrics['f1_macro']:.4f} / "
      f"{test_metrics['f1_samples']:.4f}")
print("[TEST] Precision / Recall (micro): "
      f"{test_metrics['precision_micro']:.4f} / "
      f"{test_metrics['recall_micro']:.4f}")

# Per-class F1
per_class_f1_test = f1_score(Y_test, Y_pred_test, average=None, zero_division=0)
per_class_dataframe_test = pd.DataFrame([per_class_f1_test], columns=label_list)

# Save test metrics & artifacts
pd.DataFrame([test_metrics]).to_csv(
    CHECKPOINT_DIRECTORY / "test_metrics_epoch15.csv", index=False
)
per_class_dataframe_test.to_csv(
    CHECKPOINT_DIRECTORY / "test_per_class_f1_epoch15.csv", index=False
)
np.save(CHECKPOINT_DIRECTORY / "test_y_true_epoch15.npy", Y_test)
np.save(CHECKPOINT_DIRECTORY / "test_scores_epoch15.npy", S_test)
np.save(CHECKPOINT_DIRECTORY / "test_y_pred_epoch15.npy", Y_pred_test)
with open(CHECKPOINT_DIRECTORY / "test_thresholds_used_epoch15.json", "w") as f:
    json.dump({k: float(v) for k, v in zip(label_list, T_test.flatten().tolist())}, f, indent=2)


print("\n>>>Training Finished!\n")