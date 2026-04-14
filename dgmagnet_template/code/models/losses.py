"""
Combined loss for DG-MAGNet:
    L = L_CE + alpha_adv * L_adv + alpha_mi * L_MI + alpha_reg * L_reg

- L_CE:  standard cross entropy with label smoothing
- L_adv: adversarial subject classification via Gradient Reversal Layer
- L_MI:  CLUB upper bound on I(H_inv; H_spec)
- L_reg: Frobenius norm of A_spec
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---- Gradient Reversal Layer (copied from DANN / TLL) ----

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    return GradReverse.apply(x, lambd)


class SubjectDiscriminator(nn.Module):
    """MLP that predicts subject id from features. Gets reversed gradient."""
    def __init__(self, d_in: int, num_subjects: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_subjects),
        )

    def forward(self, features, lambd: float = 1.0):
        reversed_features = grad_reverse(features, lambd)
        return self.net(reversed_features)


# ---- CLUB MI upper bound (from Cheng et al. ICML 2020) ----

class CLUBSample(nn.Module):
    """
    Sample-based CLUB estimator of I(X; Y) as an upper bound.
    Used to minimize MI between H_inv and H_spec.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden: int = 128):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, y_dim),
            nn.Tanh(),
        )

    def get_mu_logvar(self, x):
        return self.p_mu(x), self.p_logvar(x)

    def forward(self, x_samples, y_samples):
        """Returns MI upper bound (scalar, to be minimized)."""
        mu, logvar = self.get_mu_logvar(x_samples)

        # log q(y|x) for positive pairs
        positive = -(mu - y_samples) ** 2 / 2.0 / logvar.exp()

        # log q(y'|x) for random shuffled negative pairs
        prediction = mu.unsqueeze(1)      # (B, 1, d)
        y_samples_expand = y_samples.unsqueeze(0)   # (1, B, d)
        negative = -((y_samples_expand - prediction) ** 2).mean(dim=1) / 2.0 / logvar.exp()

        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound

    def learning_loss(self, x_samples, y_samples):
        """Training loss for the CLUB estimator itself (likelihood)."""
        mu, logvar = self.get_mu_logvar(x_samples)
        return -((-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=-1).mean())


# ---- Combined loss module ----

class DGMAGNetLoss(nn.Module):
    """
    Orchestrates all loss components. The training loop should:
      1. Forward model -> get logits, features, aux
      2. Compute CE loss on logits
      3. Compute adv loss via discriminator (with GRL)
      4. Compute MI loss via CLUB
      5. Compute reg loss from isgd.spec_frobenius_reg(subject_ids)
      6. Combine and backward
    """
    def __init__(self, cfg: dict, d_feature: int, num_train_subjects: int):
        super().__init__()
        lcfg = cfg["losses"]
        self.cfg = lcfg

        self.ce = nn.CrossEntropyLoss(
            label_smoothing=lcfg.get("label_smoothing", 0.0)
        )

        self.use_adv = lcfg["use_adv"]
        self.use_mi = lcfg["use_mi"]
        self.use_reg = lcfg["use_reg"]

        if self.use_adv:
            self.discriminator = SubjectDiscriminator(
                d_in=d_feature,
                num_subjects=num_train_subjects,
            )
            self.grl_lambda = lcfg["adv_grl_lambda"]

        if self.use_mi:
            self.club = CLUBSample(x_dim=d_feature, y_dim=d_feature)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        subject_ids: torch.Tensor,
        isgd_module,
        h_inv: torch.Tensor | None = None,
        h_spec: torch.Tensor | None = None,
        grl_lambda: float | None = None,
    ) -> dict:
        """
        Returns:
            dict with "total" and per-component losses

        Args:
            grl_lambda: overrides self.grl_lambda when provided (used for annealing)
        """
        out = {}

        # 1) CE
        out["ce"] = self.ce(logits, labels)
        total = self.cfg["ce_weight"] * out["ce"]

        # 2) Adversarial — use annealed lambda if provided
        if self.use_adv:
            lambd = grl_lambda if grl_lambda is not None else self.grl_lambda
            subj_logits = self.discriminator(features, lambd=lambd)
            out["adv"] = F.cross_entropy(subj_logits, subject_ids)
            total = total + self.cfg["alpha_adv"] * out["adv"]

        # 3) MI (CLUB)
        if self.use_mi and h_inv is not None and h_spec is not None:
            out["mi"] = self.club(h_inv, h_spec)
            total = total + self.cfg["alpha_mi"] * out["mi"]

        # 4) Frobenius reg
        if self.use_reg:
            out["reg"] = isgd_module.spec_frobenius_reg(subject_ids)
            total = total + self.cfg["alpha_reg"] * out["reg"]

        out["total"] = total
        return out

    def club_update_step(self, h_inv: torch.Tensor, h_spec: torch.Tensor) -> torch.Tensor:
        """Separate training step for the CLUB estimator (maximize log-likelihood)."""
        if not self.use_mi:
            return torch.tensor(0.0)
        return self.club.learning_loss(h_inv, h_spec)
