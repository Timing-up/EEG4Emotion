"""
DG-MAGNet: full model combining
  1. Multi-branch spatial encoder with ISGD adjacencies
  2. Mamba-Attention temporal hybrid
  3. Classification head

Input shape: (B, S, C, F) where
    B = batch, S = sequence length (time windows),
    C = channels (62), F = frequency bands (5)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .isgd import ISGDAdjacency
from .temporal import MambaAttentionHybrid


def cheby_conv(x, A, thetas, K):
    """
    ChebyNet spectral convolution.
    Args:
        x: (B, C, F_in) node features
        A: (B, C, C) or (C, C) adjacency (batched or shared)
        thetas: list of K parameters, each (F_in, F_out)
        K: polynomial order
    Returns:
        (B, C, F_out)
    """
    if A.dim() == 2:
        A = A.unsqueeze(0).expand(x.shape[0], -1, -1)
    # Normalized Laplacian: L_tilde = 2L/lam_max - I, approximate lam_max=2
    # -> L_tilde = L - I = -D^{-1/2} A D^{-1/2}
    D = A.sum(dim=-1).clamp(min=1e-6)
    D_inv_sqrt = D.pow(-0.5)
    A_norm = A * D_inv_sqrt.unsqueeze(-1) * D_inv_sqrt.unsqueeze(-2)
    L_tilde = -A_norm  # simplified

    Tx_0 = x
    out = Tx_0 @ thetas[0]
    if K > 1:
        Tx_1 = torch.bmm(L_tilde, x)
        out = out + Tx_1 @ thetas[1]
        for k in range(2, K):
            Tx_k = 2 * torch.bmm(L_tilde, Tx_1) - Tx_0
            out = out + Tx_k @ thetas[k]
            Tx_0, Tx_1 = Tx_1, Tx_k
    return out


class ChebyBranch(nn.Module):
    """Stack of ChebyNet layers (single branch)."""
    def __init__(self, in_dim, hidden_dim, num_layers, K=4, dropout=0.3):
        super().__init__()
        self.K = K
        self.num_layers = num_layers
        self.thetas = nn.ParameterList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for l in range(num_layers):
            layer_thetas = nn.ParameterList([
                nn.Parameter(torch.randn(dims[l], dims[l+1]) * 0.1)
                for _ in range(K)
            ])
            self.thetas.append(layer_thetas)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        for l in range(self.num_layers):
            x = cheby_conv(x, A, list(self.thetas[l]), self.K)
            x = F.relu(x)
            x = self.dropout(x)
        return x


class MultiBranchSpatialEncoder(nn.Module):
    """
    4 branches: A1(learnable), A2(learnable), A_local(fixed), residual (linear).
    Fuses via softmax-gated learnable weights.
    """
    def __init__(
        self,
        num_channels: int,
        num_bands: int,
        num_subjects: int,
        d_g: int,
        K_cheb: int,
        L1: int,
        L2: int,
        A_local: torch.Tensor,     # (C, C) fixed adjacency
        low_rank_r: int,
        subject_embed_dim: int,
        isgd_enabled: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.C = num_channels
        self.F = num_bands
        self.d_g = d_g
        self.isgd_enabled = isgd_enabled

        # ISGD adjacency manager for branches A1 and A2
        self.isgd = ISGDAdjacency(
            num_channels=num_channels,
            num_branches=2,
            num_subjects=num_subjects,
            low_rank_r=low_rank_r,
            subject_embed_dim=subject_embed_dim,
        )

        # Fixed local adjacency (buffer, not learnable)
        self.register_buffer("A_local", A_local)

        # Four branches
        self.branch1 = ChebyBranch(num_bands, d_g, L1, K_cheb, dropout)
        self.branch2 = ChebyBranch(num_bands, d_g, L2, K_cheb, dropout)
        self.branch_local = ChebyBranch(num_bands, d_g, L1, K_cheb, dropout)
        self.branch_res = nn.Sequential(
            nn.Linear(num_bands, d_g),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable fusion weights over 4 branches
        self.fusion_logits = nn.Parameter(torch.zeros(4))

        # Graph-to-token: linear projection from (C * d_g) to d_g
        self.g2t = nn.Linear(num_channels * d_g, d_g)

    def forward(self, x, subject_ids):
        """
        Args:
            x: (B, S, C, F) — sequence of feature matrices
            subject_ids: (B,) long tensor
        Returns:
            tokens: (B, S, d_g)
            aux: dict with intermediate tensors for losses/visualization
        """
        B, S, C, Fb = x.shape
        x_flat = x.reshape(B * S, C, Fb)

        # Get per-subject adjacencies: (B, 2, C, C) — repeat for time steps
        if self.isgd_enabled:
            A_batch = self.isgd(subject_ids)  # (B, 2, C, C)
        else:
            A_inv = self.isgd.get_invariant()  # (2, C, C)
            A_batch = A_inv.unsqueeze(0).expand(B, -1, -1, -1)

        A1 = A_batch[:, 0].repeat_interleave(S, dim=0)  # (B*S, C, C)
        A2 = A_batch[:, 1].repeat_interleave(S, dim=0)

        # Four branches
        h1 = self.branch1(x_flat, A1)                           # (B*S, C, d_g)
        h2 = self.branch2(x_flat, A2)
        h_local = self.branch_local(x_flat, self.A_local)       # fixed A
        h_res = self.branch_res(x_flat)                         # (B*S, C, d_g)

        # Softmax-gated fusion — keep ISGD and fixed contributions separate for MI
        alpha = F.softmax(self.fusion_logits, dim=0)
        h_isgd  = alpha[0] * h1 + alpha[1] * h2           # subject-specific branches
        h_fixed = alpha[2] * h_local + alpha[3] * h_res   # invariant branches
        h_fused = h_isgd + h_fixed                         # (B*S, C, d_g)

        # Pooled representations for CLUB MI loss (no extra forward pass):
        #   h_spec: from ISGD-modulated branches (subject-specific)
        #   h_inv:  from fixed branches (anatomically invariant)
        # Pool over channels → (B*S, d_g) → (B, S, d_g) → mean over S → (B, d_g)
        h_spec_pooled = h_isgd.mean(dim=1).reshape(B, S, self.d_g).mean(dim=1)
        h_inv_pooled  = h_fixed.mean(dim=1).reshape(B, S, self.d_g).mean(dim=1)

        # Graph-to-token: flatten channel dim, project to d_g
        tokens = self.g2t(h_fused.reshape(B * S, -1))           # (B*S, d_g)
        tokens = tokens.reshape(B, S, self.d_g)

        aux = {
            "A_batch": A_batch,
            "alpha": alpha,
            "h_fused": h_fused.reshape(B, S, C, self.d_g),
            "h_inv_pooled": h_inv_pooled,    # (B, d_g) — fixed-branch features
            "h_spec_pooled": h_spec_pooled,  # (B, d_g) — ISGD-branch features
        }
        return tokens, aux


class DGMAGNet(nn.Module):
    """Full DG-MAGNet model."""
    def __init__(self, cfg: dict, A_local: torch.Tensor, num_subjects: int):
        super().__init__()
        self.cfg = cfg
        mcfg = cfg["model"]
        icfg = cfg["isgd"]

        self.spatial = MultiBranchSpatialEncoder(
            num_channels=mcfg["num_channels"] if "num_channels" in mcfg else cfg["data"]["num_channels"],
            num_bands=cfg["data"]["num_bands"],
            num_subjects=num_subjects,
            d_g=mcfg["d_g"],
            K_cheb=mcfg["K_cheb"],
            L1=mcfg["L1"],
            L2=mcfg["L2"],
            A_local=A_local,
            low_rank_r=icfg["low_rank_r"],
            subject_embed_dim=icfg["subject_embed_dim"],
            isgd_enabled=icfg["enabled"],
            dropout=mcfg["dropout"],
        )

        self.temporal = MambaAttentionHybrid(
            d_model=mcfg["d_g"],
            depth=mcfg["mamba_depth"] + 1,  # +1 for final attention block
            d_state=mcfg["d_state"],
            mamba_expand=mcfg["mamba_expand"],
            num_heads=mcfg["attention_heads"],
            head_dim=mcfg["attention_head_dim"],
            sta_kernel=mcfg["sta_kernel"],
            ffn_ratio=mcfg["ffn_ratio"],
            dropout=mcfg["dropout"],
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(mcfg["d_g"]),
            nn.Linear(mcfg["d_g"], mcfg["num_classes"]),
        )

    def forward(self, x, subject_ids, return_features=False):
        """
        x: (B, S, C, F)
        subject_ids: (B,)
        """
        tokens, aux = self.spatial(x, subject_ids)     # (B, S, d_g)
        z = self.temporal(tokens)                      # (B, S, d_g)
        z_pooled = z.mean(dim=1)                       # (B, d_g)
        logits = self.classifier(z_pooled)             # (B, num_classes)

        if return_features:
            return logits, z_pooled, aux
        return logits

    def forward_with_external_embedding(self, x, external_embed):
        """Used during TTGA. external_embed: (dE,) learnable parameter."""
        B, S, C, Fb = x.shape
        # Build adjacencies from external embedding
        A_batch = self.spatial.isgd.forward_with_external_embedding(external_embed)
        # (1, 2, C, C) -> expand to batch
        A_batch = A_batch.expand(B, -1, -1, -1)

        x_flat = x.reshape(B * S, C, Fb)
        A1 = A_batch[:, 0].repeat_interleave(S, dim=0)
        A2 = A_batch[:, 1].repeat_interleave(S, dim=0)

        h1 = self.spatial.branch1(x_flat, A1)
        h2 = self.spatial.branch2(x_flat, A2)
        h_local = self.spatial.branch_local(x_flat, self.spatial.A_local)
        h_res = self.spatial.branch_res(x_flat)

        alpha = F.softmax(self.spatial.fusion_logits, dim=0)
        h_fused = alpha[0]*h1 + alpha[1]*h2 + alpha[2]*h_local + alpha[3]*h_res
        tokens = self.spatial.g2t(h_fused.reshape(B*S, -1)).reshape(B, S, -1)
        z = self.temporal(tokens)
        z_pooled = z.mean(dim=1)
        logits = self.classifier(z_pooled)
        return logits
