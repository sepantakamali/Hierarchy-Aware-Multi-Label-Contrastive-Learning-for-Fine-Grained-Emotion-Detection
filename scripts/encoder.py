import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoder(nn.Module):
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.projection_dim = projection_dim
        self.output_dim = projection_dim

        # Projection head: 2-layer MLP with ReLU, dropout, and normalisation
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, projection_dim),
            nn.LayerNorm(projection_dim)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        # Get token embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
        last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)

        # Mean pooling over valid tokens (mask-aware)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # shape: (batch_size, hidden_size)

        # Project into contrastive embedding space
        projected = self.projector(mean_pooled)  # shape: (batch_size, projection_dim)
        normalized = F.normalize(projected, dim=-1)  # L2 normalise for cosine similarity

        return normalized  # final contrastive embeddings
