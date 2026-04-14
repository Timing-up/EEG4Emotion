"""
Test-Time Graph Adaptation (TTGA)
==================================

At test time, we only adapt a newly-instantiated subject embedding for
the unseen target subject. All other model parameters are frozen.
Optimization objective: entropy minimization on unlabeled target samples
with Frobenius regularization to prevent drift.

Usage:
    model.eval()
    target_embed = ttga_adapt(
        model, target_unlabeled_samples,
        steps=5, lr=1e-3,
    )
    with torch.no_grad():
        logits = model.forward_with_external_embedding(test_x, target_embed)
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax(logits), averaged over batch."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


def ttga_adapt(
    model: nn.Module,
    target_samples: torch.Tensor,
    steps: int = 5,
    lr: float = 1e-3,
    entropy_weight: float = 1.0,
    frobenius_weight: float = 0.1,
    init: str = "mean",
    verbose: bool = False,
) -> nn.Parameter:
    """
    Adapt a new subject embedding on unlabeled target samples.

    Args:
        model: trained DGMAGNet
        target_samples: (N, S, C, F) unlabeled samples from the target subject
        steps: number of adaptation iterations
        lr: adaptation learning rate
        entropy_weight: weight on entropy minimization loss
        frobenius_weight: weight on ||target_embed||^2 regularization
        init: "mean" | "random" — how to initialize the new embedding
        verbose: print per-step loss

    Returns:
        target_embed: adapted nn.Parameter of shape (dE,)
    """
    # Freeze everything in the model
    for p in model.parameters():
        p.requires_grad = False

    # Create a fresh embedding for the target subject
    isgd_module = model.spatial.isgd
    target_embed = isgd_module.new_target_embedding(init=init)
    target_embed = target_embed.to(target_samples.device)
    target_embed.requires_grad = True

    optimizer = torch.optim.Adam([target_embed], lr=lr)

    # Save original model mode and set to eval
    was_training = model.training
    model.eval()

    for step in range(steps):
        optimizer.zero_grad()
        logits = model.forward_with_external_embedding(target_samples, target_embed)
        ent = entropy_loss(logits)
        frob = (target_embed ** 2).sum()
        loss = entropy_weight * ent + frobenius_weight * frob
        loss.backward()
        optimizer.step()

        if verbose:
            print(f"  [TTGA step {step+1}/{steps}] "
                  f"ent={ent.item():.4f} frob={frob.item():.4f} "
                  f"loss={loss.item():.4f}")

    # Restore model mode
    if was_training:
        model.train()

    # Unfreeze model (restore requires_grad)
    for p in model.parameters():
        p.requires_grad = True

    target_embed.requires_grad = False
    return target_embed


@torch.no_grad()
def ttga_predict(
    model: nn.Module,
    test_samples: torch.Tensor,
    target_embed: torch.Tensor,
) -> torch.Tensor:
    """Run inference using the adapted target embedding."""
    model.eval()
    logits = model.forward_with_external_embedding(test_samples, target_embed)
    return logits


def ttga_full_pipeline(
    model: nn.Module,
    target_unlabeled: torch.Tensor,
    target_test: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """
    Full TTGA pipeline: adapt on unlabeled, predict on test.
    Called once per target subject in LOSO evaluation.

    Args:
        target_unlabeled: small set of unlabeled target samples for adaptation
        target_test: full target test set for final prediction
        cfg: ttga section of config
    """
    target_embed = ttga_adapt(
        model,
        target_unlabeled,
        steps=cfg["adapt_steps"],
        lr=cfg["adapt_lr"],
        entropy_weight=cfg["entropy_weight"],
        frobenius_weight=cfg["frobenius_weight"],
        init=cfg.get("init_from", "mean").replace("mean_train_embedding", "mean"),
    )
    logits = ttga_predict(model, target_test, target_embed)
    return logits
