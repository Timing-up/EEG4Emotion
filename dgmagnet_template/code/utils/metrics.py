"""
Evaluation metrics for DG-MAGNet
================================

- Classification: accuracy, macro-F1, weighted-F1, per-class precision/recall
- Statistical: Wilcoxon signed-rank test for pairwise ablation comparison
- Summary table formatting for LOSO results
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> dict:
    """Compute full classification metrics.

    Returns dict with accuracy, macro_f1, weighted_f1, per-class P/R/F1,
    and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)

    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    labels = list(range(num_classes)) if num_classes else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "per_class_precision": per_class_p.tolist(),
        "per_class_recall": per_class_r.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# LOSO aggregation
# ---------------------------------------------------------------------------

def aggregate_loso_metrics(fold_metrics: list[dict]) -> dict:
    """Aggregate per-fold metric dicts into mean +/- std summary."""
    keys = ["accuracy", "macro_f1", "weighted_f1"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"] = float(np.std(vals))
        summary[f"per_fold_{k}"] = vals
    return summary


def format_loso_table(results: dict[str, dict], metric: str = "accuracy") -> str:
    """Format ablation results as a printable table.

    Args:
        results: {variant_name: aggregate_dict} from aggregate_loso_metrics
        metric: which metric to display
    """
    lines = [f"{'Variant':<20} {'Mean':>8} {'Std':>8}"]
    lines.append("-" * 38)
    for name, r in results.items():
        mean_key = f"mean_{metric}"
        std_key = f"std_{metric}"
        lines.append(f"{name:<20} {r[mean_key]:8.3f} {r[std_key]:8.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_test(
    scores_a: list[float],
    scores_b: list[float],
    alternative: str = "greater",
) -> dict:
    """Wilcoxon signed-rank test comparing two matched sets of fold scores.

    Args:
        scores_a: per-fold scores for method A (expected better)
        scores_b: per-fold scores for method B (baseline)
        alternative: "greater" tests if A > B

    Returns:
        dict with statistic, p_value, significant (at alpha=0.05), and effect_size
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diffs = a - b

    # Remove zero differences (ties)
    nonzero = diffs != 0
    if nonzero.sum() < 6:
        return {
            "statistic": None,
            "p_value": None,
            "significant": False,
            "note": "Too few non-tied pairs for reliable Wilcoxon test",
            "mean_diff": float(diffs.mean()),
        }

    stat, p = stats.wilcoxon(a, b, alternative=alternative)

    # Effect size r = Z / sqrt(N)
    n = nonzero.sum()
    z = stats.norm.ppf(1 - p / 2)  # approximate Z from p
    effect_r = z / np.sqrt(n)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < 0.05,
        "effect_size_r": float(effect_r),
        "mean_diff": float(diffs.mean()),
    }


def pairwise_wilcoxon(
    results: dict[str, list[float]],
    baseline: str | None = None,
) -> dict:
    """Run Wilcoxon tests for all variants against a baseline.

    Args:
        results: {variant_name: per_fold_scores}
        baseline: name of baseline variant (defaults to first key)

    Returns:
        dict of {variant: wilcoxon_result}
    """
    names = list(results.keys())
    if baseline is None:
        baseline = names[0]
    base_scores = results[baseline]

    comparisons = {}
    for name in names:
        if name == baseline:
            continue
        comparisons[f"{name}_vs_{baseline}"] = wilcoxon_test(
            results[name], base_scores
        )
    return comparisons
