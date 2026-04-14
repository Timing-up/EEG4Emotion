"""
Temporal module: Mamba + Attention hybrid for EEG token sequences.

Input:  (B, S, d_g) token sequence from spatial encoder
Output: (B, S, d_g) temporally contextualized sequence
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("[WARN] mamba-ssm not installed. Using fallback GRU.")


class MambaBlockFallback(nn.Module):
    """Fallback if mamba-ssm unavailable: bi-GRU with same I/O shape."""
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True, bidirectional=True)
    def forward(self, x):
        out, _ = self.gru(x)
        return out


class FFN(nn.Module):
    def __init__(self, d: int, ratio: float = 1.0, dropout: float = 0.3):
        super().__init__()
        hidden = int(d * max(1.0, ratio) * 2)
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class PreNormResidual(nn.Module):
    """Standard PreNorm + residual wrapper."""
    def __init__(self, d: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fn = fn
    def forward(self, x, **kwargs):
        return x + self.fn(self.norm(x), **kwargs)


class AttentionSTA(nn.Module):
    """
    Multi-head self-attention + Short-Time Aggregation (Conv1D after attention).
    Matches MAGNet paper's attention module.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        sta_kernel: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert d_model == num_heads * head_dim, \
            f"d_model({d_model}) != heads({num_heads}) * head_dim({head_dim})"
        self.H = num_heads
        self.d_h = head_dim
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        # STA: depthwise 1D conv per head
        self.sta_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=sta_kernel,
            padding=sta_kernel // 2,
            groups=num_heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.H, self.d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, d_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.d_h ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, H, S, d_h)
        out = out.transpose(1, 2).reshape(B, S, self.d_model)  # (B, S, d_model)

        # STA: apply Conv1D along sequence
        out_t = out.transpose(1, 2)                  # (B, d_model, S)
        out_t = self.sta_conv(out_t)
        out = out_t.transpose(1, 2)                  # (B, S, d_model)

        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out


class MambaAttentionHybrid(nn.Module):
    """
    Stack of (depth-1) Mamba+FFN blocks, followed by 1 Attention+FFN block.
    Matches MAGNet figure: 4x Mamba then Attention.
    """
    def __init__(
        self,
        d_model: int,
        depth: int = 5,
        d_state: int = 16,
        mamba_expand: int = 2,
        num_heads: int = 8,
        head_dim: int = 8,
        sta_kernel: int = 3,
        ffn_ratio: float = 1.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert depth >= 2, "depth must be at least 2 (1 Mamba + 1 Attention)"

        self.layers = nn.ModuleList()
        MambaCls = Mamba if HAS_MAMBA else MambaBlockFallback

        # (depth - 1) Mamba+FFN blocks
        for _ in range(depth - 1):
            mamba = MambaCls(d_model=d_model, d_state=d_state, expand=mamba_expand) \
                if HAS_MAMBA else MambaCls(d_model, d_state, mamba_expand)
            self.layers.append(PreNormResidual(d_model, mamba))
            self.layers.append(PreNormResidual(d_model, FFN(d_model, ffn_ratio, dropout)))

        # 1 Attention+FFN block
        attn = AttentionSTA(d_model, num_heads, head_dim, sta_kernel, dropout)
        self.layers.append(PreNormResidual(d_model, attn))
        self.layers.append(PreNormResidual(d_model, FFN(d_model, ffn_ratio, dropout)))

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


if __name__ == "__main__":
    m = MambaAttentionHybrid(d_model=64, depth=5)
    x = torch.randn(4, 9, 64)
    y = m(x)
    print(f"In: {x.shape}, Out: {y.shape}")
    assert y.shape == x.shape
    print("Temporal module OK ✓")
