import os
# Enable deterministic CuBLAS operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

@torch.no_grad()
def initialize_prototypes(dataloader, encoder, label_list, device="cuda"):
    """
    Initialize prototypes by averaging the embeddings of all samples per label.

    Args:
        dataloader: DataLoader object yielding batches with 'text' and 'labels'
        encoder: Model that outputs embeddings given input text
        label_list: List of all class labels
        device: Torch device

    Returns:
        Tensor: Prototype tensor of shape [C, D] where C is number of classes and D is embedding dimension.
    """
    encoder.eval()
    device = device or "cuda"
   
    prototypes_sum = None
    prototypes_count = None

    for batch in tqdm(dataloader, desc="Initializing prototypes", total=len(dataloader) if hasattr(dataloader, '__len__') else None):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)            # [B×C]
        embeddings = encoder(input_ids=input_ids, attention_mask=attention_mask)  # [B×D]
        B, D = embeddings.shape
        C = labels.size(1)
        # Initialize...
        if prototypes_sum is None:
            prototypes_sum = torch.zeros(C, D, device=device)
            prototypes_count = torch.zeros(C, device=device)
        # Accumulate sums and counts
        prototypes_sum += labels.T @ embeddings   # [C×D]
        prototypes_count += labels.sum(dim=0)     # [C]

    prototypes = prototypes_sum / prototypes_count.clamp(min=1).unsqueeze(1)  # [C×D]
    prototypes = F.normalize(prototypes, dim=1)
    return prototypes


@torch.no_grad()
def update_prototypes(prototypes, embeddings, multi_hot_labels, momentum=0.1):
    """
    Update prototypes using exponential moving average (EMA).

    Args:
        prototypes (Tensor): Prototype tensor of shape [C, D]
        embeddings (Tensor): Batch embeddings, shape [batch_size, embedding_dim]
        multi_hot_labels (Tensor): Multi-hot labels, shape [batch_size, num_classes]
        momentum (float): Momentum factor for EMA

    Returns:
        Tensor: Updated prototype tensor of shape [C, D]
    """
    embeddings = embeddings.to(prototypes.device)
    multi_hot_labels = multi_hot_labels.to(prototypes.device).float()  # [B×C]

    batch_sum = multi_hot_labels.T @ embeddings                  # [C×D]
    batch_count = multi_hot_labels.sum(dim=0)                    # [C]

    updated_prototypes = prototypes.clone()
    mask = batch_count > 0
    if mask.any():
        batch_average = torch.zeros_like(prototypes)
        batch_average[mask] = batch_sum[mask] / batch_count[mask].unsqueeze(1)
        batch_average = F.normalize(batch_average, dim=1)  # Normalize directions before mixing
        updated_prototypes[mask] = (1 - momentum) * prototypes[mask] + (momentum) * batch_average[mask]

    return F.normalize(updated_prototypes, dim=1)