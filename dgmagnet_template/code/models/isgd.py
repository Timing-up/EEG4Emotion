"""
ISGD: Invariant-Specific Graph Decomposition
=============================================

Core idea: decompose each learnable adjacency matrix into a globally shared
invariant part and a subject-specific low-rank modulation:

    A^k(s) = A_inv^k + lambda * A_spec^k(s)
    A_spec^k(s) = U_s @ V_s.T    (rank r)
    U_s, V_s = MLP_k(subject_embed(s))

At inference on unseen subjects, initialize subject_embed from the mean
of training subject embeddings, then optionally adapt via TTGA.

Shape conventions:
    C  = num_channels (62 for SEED, 32 for DEAP)
    r  = low_rank_r
    dE = subject_embed_dim
    S  = num_train_subjects (varies per LOSO fold)
    K  = number of learnable branches (default 2: A1, A2)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ISGDAdjacency(nn.Module):
    """
    Maintains K parallel (A_inv, A_spec) pairs, one per learnable branch.

    Args:
        num_channels: C
        num_branches: K (typically 2)
        num_subjects: size of training subject pool for this LOSO fold
        low_rank_r: rank of A_spec low-rank factorization
        subject_embed_dim: dE
        lambda_spec: weight on specific component at forward time
        symmetric: if True, returns symmetric non-negative adjacency
    """

    def __init__(
        self,
        num_channels: int,
        num_branches: int,
        num_subjects: int,
        low_rank_r: int = 8,
        subject_embed_dim: int = 32,
        lambda_spec: float = 1.0,
        symmetric: bool = True,
    ):
        super().__init__()
        self.C = num_channels
        self.K = num_branches
        self.r = low_rank_r
        self.dE = subject_embed_dim
        self.lambda_spec = lambda_spec
        self.symmetric = symmetric

        # Invariant adjacencies: one per branch, shared across all subjects
        # Parameterize as raw unconstrained tensor; symmetrize + ReLU at forward
        self.A_inv_raw = nn.Parameter(
            torch.randn(num_branches, num_channels, num_channels) * 0.01
        )

        # Subject embedding table
        self.subject_embed = nn.Embedding(num_subjects, subject_embed_dim)
        nn.init.normal_(self.subject_embed.weight, std=0.02)

        # One MLP per branch that generates U and V from subject embedding
        # Output: [U (C*r) | V (C*r)] -> 2*C*r
        self.uv_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(subject_embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, 2 * num_channels * low_rank_r),
            )
            for _ in range(num_branches)
        ])

    def _symmetrize(self, A: torch.Tensor) -> torch.Tensor:
        """ReLU(A + A^T) — matches original MAGNet convention."""
        return F.relu(A + A.transpose(-1, -2))

    def get_invariant(self) -> torch.Tensor:
        """Returns symmetrized invariant adjacencies, shape (K, C, C)."""
        return self._symmetrize(self.A_inv_raw)

    def get_specific(self, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute A_spec for a batch of subjects.
        Args:
            subject_ids: (B,) long tensor of subject indices
        Returns:
            A_spec: (B, K, C, C)
        """
        B = subject_ids.shape[0]
        embeds = self.subject_embed(subject_ids)  # (B, dE)

        specs = []
        for k in range(self.K):
            uv = self.uv_generators[k](embeds)  # (B, 2*C*r)
            uv = uv.view(B, 2, self.C, self.r)
            U = uv[:, 0]  # (B, C, r)
            V = uv[:, 1]  # (B, C, r)
            A_spec_k = torch.matmul(U, V.transpose(-1, -2))  # (B, C, C)
            if self.symmetric:
                A_spec_k = self._symmetrize(A_spec_k)
            specs.append(A_spec_k)

        return torch.stack(specs, dim=1)  # (B, K, C, C)

    def forward(
        self,
        subject_ids: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Compose A(s) = A_inv + lambda * A_spec(s) for each subject in batch.

        Args:
            subject_ids: (B,) long tensor
            return_parts: if True, also return (A_inv, A_spec) separately
        Returns:
            A: (B, K, C, C) composed adjacencies
            (optional) A_inv: (K, C, C), A_spec: (B, K, C, C)
        """
        A_inv = self.get_invariant()                    # (K, C, C)
        A_spec = self.get_specific(subject_ids)          # (B, K, C, C)
        A = A_inv.unsqueeze(0) + self.lambda_spec * A_spec

        if return_parts:
            return A, A_inv, A_spec
        return A

    def spec_frobenius_reg(self, subject_ids: torch.Tensor) -> torch.Tensor:
        """Frobenius norm regularization on A_spec — prevents drift."""
        A_spec = self.get_specific(subject_ids)
        return (A_spec ** 2).sum(dim=(-1, -2)).mean()

    # ---- Inference-time utilities ----

    def mean_train_embedding(self) -> torch.Tensor:
        """Returns mean of all training subject embeddings, shape (dE,)."""
        return self.subject_embed.weight.mean(dim=0).detach()

    def new_target_embedding(self, init: str = "mean") -> nn.Parameter:
        """
        Create a fresh learnable embedding for an unseen target subject.
        Used by TTGA at test time.
        """
        if init == "mean":
            init_vec = self.mean_train_embedding()
        elif init == "random":
            init_vec = torch.randn(self.dE) * 0.02
        else:
            raise ValueError(f"Unknown init: {init}")
        return nn.Parameter(init_vec.clone())

    def forward_with_external_embedding(
        self, external_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward with an externally-supplied embedding (used during TTGA).
        Args:
            external_embed: (dE,) or (B, dE)
        Returns:
            A: (1, K, C, C) or (B, K, C, C)
        """
        if external_embed.dim() == 1:
            external_embed = external_embed.unsqueeze(0)  # (1, dE)
        B = external_embed.shape[0]

        A_inv = self.get_invariant()
        specs = []
        for k in range(self.K):
            uv = self.uv_generators[k](external_embed)
            uv = uv.view(B, 2, self.C, self.r)
            U, V = uv[:, 0], uv[:, 1]
            A_spec_k = torch.matmul(U, V.transpose(-1, -2))
            if self.symmetric:
                A_spec_k = self._symmetrize(A_spec_k)
            specs.append(A_spec_k)
        A_spec = torch.stack(specs, dim=1)

        return A_inv.unsqueeze(0) + self.lambda_spec * A_spec


# ---- Unit test ----
if __name__ == "__main__":
    torch.manual_seed(0)
    C, K, S = 62, 2, 14
    isgd = ISGDAdjacency(
        num_channels=C, num_branches=K, num_subjects=S,
        low_rank_r=8, subject_embed_dim=32,
    )
    sids = torch.randint(0, S, (4,))
    A = isgd(sids)
    print(f"A shape: {A.shape}")           # (4, 2, 62, 62)
    assert A.shape == (4, K, C, C)
    assert (A >= 0).all(), "ReLU should give non-negative"
    # Symmetry check
    assert torch.allclose(A, A.transpose(-1, -2), atol=1e-5)
    # Grad flow
    loss = A.sum() + isgd.spec_frobenius_reg(sids)
    loss.backward()
    assert isgd.A_inv_raw.grad is not None
    assert isgd.subject_embed.weight.grad is not None
    print("ISGD unit test passed ✓")
