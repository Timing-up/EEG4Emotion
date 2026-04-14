"""
DG-MAGNet LOSO Trainer
======================

Leave-One-Subject-Out training loop with:
  - Combined loss (CE + Adv + MI + Reg)
  - Separate CLUB estimator updates
  - Mixed precision (AMP)
  - Cosine / step LR scheduling with warmup
  - Early stopping on validation macro-F1
  - Checkpoint saving / loading
  - WandB logging
  - TTGA evaluation on held-out subject

Expects a dataloader factory that returns (train_loader, val_loader, test_loader)
for each LOSO fold, with batches of (x, labels, subject_ids).
"""
from __future__ import annotations

import os
import math
import time
import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from ..models.dg_magnet import DGMAGNet
from ..models.losses import DGMAGNetLoss
from ..adaptation.ttga import ttga_full_pipeline

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_optimizer(params, cfg: dict):
    ocfg = cfg["training"]
    if ocfg["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(
            params,
            lr=ocfg["lr"],
            weight_decay=ocfg["weight_decay"],
            betas=tuple(ocfg["betas"]),
        )
    raise ValueError(f"Unknown optimizer: {ocfg['optimizer']}")


def build_scheduler(optimizer, cfg: dict):
    tcfg = cfg["training"]
    stype = tcfg.get("scheduler", "cosine")
    if stype == "cosine":
        return CosineAnnealingLR(optimizer, T_max=tcfg["epochs"] - tcfg.get("warmup_epochs", 0))
    if stype == "step":
        return StepLR(optimizer, step_size=tcfg.get("step_size", 5), gamma=0.5)
    return None


def warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return
    lr = base_lr * (epoch + 1) / warmup_epochs
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def compute_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int):
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


# ---------------------------------------------------------------------------
# Single-fold trainer
# ---------------------------------------------------------------------------

class FoldTrainer:
    """Trains one LOSO fold (all train subjects -> one test subject)."""

    def __init__(
        self,
        cfg: dict,
        model: DGMAGNet,
        loss_module: DGMAGNetLoss,
        train_loader,
        val_loader,
        test_loader,
        fold_id: int,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.loss_module = loss_module.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.fold_id = fold_id
        self.device = device

        # Optimizers
        all_params = list(model.parameters()) + list(loss_module.parameters())
        self.optimizer = build_optimizer(all_params, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)

        # Separate optimizer for CLUB estimator
        if loss_module.use_mi:
            self.club_optimizer = torch.optim.Adam(
                loss_module.club.parameters(), lr=cfg["training"]["lr"]
            )
        else:
            self.club_optimizer = None

        # AMP
        self.use_amp = cfg["training"].get("mixed_precision", False) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # GRL annealing step counter
        self.global_step = 0
        self._total_steps = cfg["training"]["epochs"] * 1000  # rough estimate; updated in first epoch

        # Early stopping
        self.patience = cfg["training"].get("early_stop_patience", 20)
        self.best_val_f1 = -1.0
        self.best_state = None
        self.wait = 0

        # Checkpointing
        self.ckpt_dir = Path(
            os.path.expanduser(cfg["logging"].get("checkpoint_dir", "outputs/checkpoints"))
        )
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training epoch ----

    def train_epoch(self, epoch: int):
        tcfg = self.cfg["training"]
        warmup_lr(self.optimizer, epoch, tcfg.get("warmup_epochs", 0), tcfg["lr"])

        self.model.train()
        self.loss_module.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)
        # Update total_steps estimate once we know the true loader length
        self._total_steps = tcfg["epochs"] * n_batches

        for batch_idx, batch in enumerate(self.train_loader):
            x, labels, subject_ids = self._unpack(batch)

            # GRL lambda annealing: λ(p) = 2/(1+exp(-10p))-1, p ∈ [0,1]
            self.global_step += 1
            p = self.global_step / max(self._total_steps, 1)
            grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

            # --- CLUB estimator update (separate step) ---
            if self.club_optimizer is not None:
                with autocast(enabled=self.use_amp):
                    _, _, aux = self.model(x, subject_ids, return_features=True)
                    h_inv, h_spec = self._get_h_inv_spec(aux)
                    club_loss = self.loss_module.club_update_step(
                        h_inv.detach(), h_spec.detach()
                    )
                self.club_optimizer.zero_grad()
                self.scaler.scale(club_loss).backward()
                self.scaler.step(self.club_optimizer)
                # scaler.update() deferred — must be called only once per batch
                # after ALL optimizer steps (see AMP docs)

            # --- Main forward + backward ---
            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):
                logits, features, aux = self.model(x, subject_ids, return_features=True)
                h_inv, h_spec = self._get_h_inv_spec(aux)
                losses = self.loss_module(
                    logits, labels, features, subject_ids,
                    self.model.spatial.isgd,
                    h_inv=h_inv, h_spec=h_spec,
                    grl_lambda=grl_lambda,
                )
                loss = losses["total"]

            self.scaler.scale(loss).backward()
            # Gradient clipping
            if tcfg.get("grad_clip", 0) > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.loss_module.parameters()),
                    tcfg["grad_clip"],
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        # Step scheduler after warmup
        if epoch >= self.cfg["training"].get("warmup_epochs", 0) and self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(n_batches, 1)

    # ---- Validation ----

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in self.val_loader:
            x, labels, subject_ids = self._unpack(batch)
            logits = self.model(x, subject_ids)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return compute_metrics(all_preds, all_labels, self.cfg["model"]["num_classes"])

    # ---- Test (with optional TTGA) ----

    def evaluate_test(self):
        # NOTE: no @torch.no_grad() here — TTGA branch needs grad computation.
        # Grad is explicitly disabled in the plain-inference path below.
        self.model.eval()
        ttga_cfg = self.cfg.get("ttga", {})

        if ttga_cfg.get("enabled", False):
            return self._evaluate_with_ttga()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                x, labels, subject_ids = self._unpack(batch)
                logits = self.model(x, subject_ids)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return compute_metrics(all_preds, all_labels, self.cfg["model"]["num_classes"])

    def _evaluate_with_ttga(self):
        ttga_cfg = self.cfg["ttga"]

        # Collect all test data
        all_x, all_y = [], []
        for batch in self.test_loader:
            x, labels, _ = self._unpack(batch)
            all_x.append(x)
            all_y.append(labels)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)

        n_total = all_x.shape[0]
        n_adapt = min(ttga_cfg.get("adapt_samples", 5), n_total // 2)

        # Strictly separate adaptation samples from evaluation samples
        # to prevent any form of test-time data leakage.
        adapt_x = all_x[:n_adapt]
        eval_x  = all_x[n_adapt:]
        eval_y  = all_y[n_adapt:]

        if eval_x.shape[0] == 0:
            # Degenerate: not enough test samples; skip adaptation
            with torch.no_grad():
                logits = self.model(all_x, torch.zeros(n_total, dtype=torch.long,
                                                        device=self.device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            return compute_metrics(preds, all_y.cpu().numpy(),
                                   self.cfg["model"]["num_classes"])

        logits = ttga_full_pipeline(self.model, adapt_x, eval_x, ttga_cfg)
        preds = logits.argmax(dim=-1).cpu().numpy()
        return compute_metrics(preds, eval_y.cpu().numpy(),
                               self.cfg["model"]["num_classes"])

    # ---- Early stopping + checkpoint ----

    def check_early_stop(self, val_metrics: dict) -> bool:
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.best_state = {
                "model": copy.deepcopy(self.model.state_dict()),
                "loss_module": copy.deepcopy(self.loss_module.state_dict()),
            }
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore_best(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["model"])
            self.loss_module.load_state_dict(self.best_state["loss_module"])

    def save_checkpoint(self, epoch: int, tag: str = "best"):
        path = self.ckpt_dir / f"fold{self.fold_id}_{tag}.pt"
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "loss_module": self.loss_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
        }, path)

    # ---- Helpers ----

    def _unpack(self, batch):
        x, labels, subject_ids = batch[0], batch[1], batch[2]
        return x.to(self.device), labels.to(self.device), subject_ids.to(self.device)

    def _get_h_inv_spec(self, aux: dict) -> tuple:
        """Return (h_inv, h_spec) from spatial encoder branch separation.

        h_inv  — pooled features from fixed (anatomically invariant) branches
        h_spec — pooled features from ISGD-modulated (subject-specific) branches
        Both shapes: (B, d_g)
        """
        h_inv  = aux.get("h_inv_pooled")
        h_spec = aux.get("h_spec_pooled")
        if h_inv is not None and h_spec is not None:
            return h_inv, h_spec
        # Fallback for older checkpoints without split features
        d_g = self.cfg["model"]["d_g"]
        h_fused = aux.get("h_fused")
        if h_fused is not None:
            pooled = h_fused.mean(dim=(1, 2))  # (B, d_g)
            return pooled, pooled
        B = next(iter(aux.values())).shape[0]
        zeros = torch.zeros(B, d_g, device=self.device)
        return zeros, zeros


# ---------------------------------------------------------------------------
# Full LOSO runner
# ---------------------------------------------------------------------------

def run_loso(
    cfg: dict,
    build_loaders_fn: Callable,
    build_A_local_fn: Callable,
    device: torch.device | None = None,
):
    """
    Run full LOSO cross-validation.

    Args:
        cfg: merged config dict
        build_loaders_fn: (cfg, fold_id) -> (train_loader, val_loader, test_loader, num_train_subjects)
        build_A_local_fn: (cfg) -> Tensor of shape (C, C) fixed local adjacency
        device: torch device (defaults to cfg["experiment"]["device"])

    Returns:
        dict with per-fold and aggregated metrics
    """
    if device is None:
        dev_str = cfg["experiment"].get("device", "cuda")
        device = torch.device(dev_str if torch.cuda.is_available() else "cpu")

    A_local = build_A_local_fn(cfg).to(device)
    n_folds = cfg["evaluation"]["n_folds"]
    seeds = cfg["evaluation"].get("seeds", [cfg["experiment"]["seed"]])

    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed_results = []

        # Init wandb run
        use_wandb = cfg["logging"].get("use_wandb", False) and HAS_WANDB
        if use_wandb:
            wandb.init(
                project=cfg["logging"].get("wandb_project", "dg-magnet"),
                entity=cfg["logging"].get("wandb_entity"),
                name=f"{cfg['experiment']['name']}_seed{seed}",
                config=cfg,
                reinit=True,
            )

        for fold in range(n_folds):
            print(f"\n{'='*60}")
            print(f"Seed {seed} | Fold {fold}/{n_folds-1} (test subject = {fold})")
            print(f"{'='*60}")

            train_loader, val_loader, test_loader, num_train_subjects = \
                build_loaders_fn(cfg, fold)

            model = DGMAGNet(cfg, A_local, num_train_subjects)
            loss_module = DGMAGNetLoss(cfg, d_feature=cfg["model"]["d_g"],
                                       num_train_subjects=num_train_subjects)

            trainer = FoldTrainer(
                cfg, model, loss_module,
                train_loader, val_loader, test_loader,
                fold_id=fold, device=device,
            )

            # Training loop
            for epoch in range(cfg["training"]["epochs"]):
                t0 = time.time()
                train_loss = trainer.train_epoch(epoch)
                val_metrics = trainer.validate()
                elapsed = time.time() - t0

                if (epoch + 1) % cfg["logging"].get("log_every", 20) == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:3d} | loss={train_loss:.4f} "
                          f"| val_acc={val_metrics['accuracy']:.3f} "
                          f"| val_f1={val_metrics['macro_f1']:.3f} "
                          f"| {elapsed:.1f}s")

                if use_wandb:
                    wandb.log({
                        f"fold{fold}/train_loss": train_loss,
                        f"fold{fold}/val_acc": val_metrics["accuracy"],
                        f"fold{fold}/val_f1": val_metrics["macro_f1"],
                        "epoch": epoch,
                    })

                if trainer.check_early_stop(val_metrics):
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            # Restore best and evaluate
            trainer.restore_best()
            if cfg["logging"].get("save_checkpoints", False):
                trainer.save_checkpoint(epoch, tag="best")

            test_metrics = trainer.evaluate_test()
            print(f"  >> Fold {fold} test: acc={test_metrics['accuracy']:.3f} "
                  f"f1={test_metrics['macro_f1']:.3f}")
            seed_results.append(test_metrics)

            if use_wandb:
                wandb.log({
                    f"fold{fold}/test_acc": test_metrics["accuracy"],
                    f"fold{fold}/test_f1": test_metrics["macro_f1"],
                })

        # Aggregate seed results
        accs = [r["accuracy"] for r in seed_results]
        f1s = [r["macro_f1"] for r in seed_results]
        seed_summary = {
            "seed": seed,
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs),
            "mean_f1": np.mean(f1s),
            "std_f1": np.std(f1s),
            "per_fold": seed_results,
        }
        all_results.append(seed_summary)

        print(f"\n--- Seed {seed} summary ---")
        print(f"  ACC: {seed_summary['mean_acc']:.3f} +/- {seed_summary['std_acc']:.3f}")
        print(f"  F1:  {seed_summary['mean_f1']:.3f} +/- {seed_summary['std_f1']:.3f}")

        if use_wandb:
            wandb.log({
                "mean_acc": seed_summary["mean_acc"],
                "mean_f1": seed_summary["mean_f1"],
            })
            wandb.finish()

    # Multi-seed aggregate
    all_accs = [s["mean_acc"] for s in all_results]
    all_f1s = [s["mean_f1"] for s in all_results]
    final = {
        "mean_acc": np.mean(all_accs),
        "std_acc": np.std(all_accs),
        "mean_f1": np.mean(all_f1s),
        "std_f1": np.std(all_f1s),
        "per_seed": all_results,
    }

    print(f"\n{'='*60}")
    print(f"FINAL ({len(seeds)} seeds): ACC={final['mean_acc']:.3f}+/-{final['std_acc']:.3f} "
          f"F1={final['mean_f1']:.3f}+/-{final['std_f1']:.3f}")
    print(f"{'='*60}")

    # Save results
    out_dir = Path(os.path.expanduser(
        cfg["logging"].get("figures_dir", "outputs/figures")
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loso_results.json", "w") as f:
        json.dump(final, f, indent=2, default=float)

    return final
