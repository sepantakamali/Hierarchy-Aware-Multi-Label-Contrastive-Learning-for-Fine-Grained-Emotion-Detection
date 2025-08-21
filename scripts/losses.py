import torch
import torch.nn as nn

class PrototypicalContrastiveLoss(nn.Module):
    def __init__(
        self,
        hierarchy_dictionary,
        embedding_dim,
        alpha: float = 1.0, # Pontrastive weight
        beta: float = 0.20, # Prototype pull weight
        temp_min: float = 0.07, # τ bounds
        temp_max: float = 0.12,
    ):
        super().__init__()
        self.logit_scale_con = nn.Parameter(torch.log(torch.tensor(0.07, dtype=torch.float32)))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)

    def forward(self, embeddings, multi_hot_labels, prototypes, epoch):
        # embeddings/prototypes are already L2-normalized
        B = embeddings.size(0)
        device = embeddings.device
        eps = 1e-9

        # pairwise cosine sim
        sim = embeddings @ embeddings.T
        tau = self.logit_scale_con.exp().clamp(self.temp_min, self.temp_max)
        scaled = sim / tau

        L = multi_hot_labels.float()              # [B, C]
        pos_mask = (L @ L.T) > 0                  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = pos_mask.masked_fill(eye, False)
        neg_mask = ~pos_mask

        exp_scaled = torch.exp(scaled)
        exp_pos = exp_scaled * pos_mask.float()
        exp_neg = exp_scaled * neg_mask.float()

        pos_sum = exp_pos.sum(dim=1)
        neg_sum = exp_neg.sum(dim=1)

        valid = (pos_mask.sum(dim=1) > 0)
        pos_sum_safe = torch.where(valid, pos_sum, torch.full_like(pos_sum, eps))
        den = pos_sum_safe + torch.where(valid, neg_sum, torch.zeros_like(neg_sum)) + eps

        contrastive = -torch.log(pos_sum_safe / den)
        contrastive = contrastive[valid].mean() if valid.any() else torch.zeros((), device=device)

        # prototype pull
        proto_sim = embeddings @ prototypes.T
        L_counts = L.sum(dim=1).clamp(min=1.0)
        pos_proto_mean = (proto_sim * L).sum(dim=1) / L_counts
        proto_pull = (1.0 - pos_proto_mean).mean()

        total = self.alpha * contrastive + self.beta * proto_pull
        return total