"""
DG-MAGNet training entry point.

Usage:
    python -m code.train --config configs/seed_loso.yaml
    python -m code.train --config configs/ablation.yaml --variant isgd_only
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml
import torch
import numpy as np

from .trainers.dg_trainer import run_loso
from .seed_loader import build_loaders_seed


def load_config(config_path: str, variant: str | None = None) -> dict:
    """Load YAML config with _base_ inheritance and optional variant overlay."""
    path = Path(config_path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Handle _base_ inheritance
    if "_base_" in cfg:
        base_path = path.parent / cfg.pop("_base_")
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f)
        # Handle nested base
        if "_base_" in base_cfg:
            base2_path = base_path.parent / base_cfg.pop("_base_")
            with open(base2_path) as f:
                base2_cfg = yaml.safe_load(f)
            base_cfg = deep_merge(base2_cfg, base_cfg)
        cfg = deep_merge(base_cfg, cfg)

    # Apply variant overrides from ablation config
    if variant and "variants" in cfg:
        variants = cfg.pop("variants")
        if variant not in variants:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(variants.keys())}")
        cfg = deep_merge(cfg, variants[variant])
    elif "variants" in cfg:
        cfg.pop("variants")

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


# Standard channel layouts
_SEED_62_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7',  'F5',  'F3',  'F1',  'FZ',  'F2',  'F4',  'F6',  'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7',  'C5',  'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',  'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7',  'P5',  'P3',  'P1',  'PZ',  'P2',  'P4',  'P6',  'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1',  'OZ',  'O2',  'CB2',
]

_DEAP_32_CHANNELS = [
    'FP1', 'AF3', 'F7',  'F3',  'FC1', 'FC5', 'T7',  'C3',
    'CP1', 'CP5', 'P7',  'P3',  'PZ',  'PO3', 'O1',  'OZ',
    'O2',  'PO4', 'P4',  'P8',  'CP6', 'CP2', 'C4',  'T8',
    'FC6', 'FC2', 'F4',  'F8',  'AF4', 'FP2', 'FZ',  'CZ',
]

# Approximate Cartesian coords (meters) for channels absent from MNE standard_1020
_EXTRA_COORDS = {
    'CB1': np.array([-0.076, -0.078, -0.048], dtype=np.float32),  # left mastoid
    'CB2': np.array([ 0.076, -0.078, -0.048], dtype=np.float32),  # right mastoid
}


def _get_electrode_coords(num_channels: int, dataset: str) -> np.ndarray | None:
    """Return (C, 3) array of real 10-20 electrode positions via MNE."""
    ds = dataset.upper().replace('-', '')
    if ds in ('SEED', 'SEEDIV') and num_channels == 62:
        ch_names = _SEED_62_CHANNELS
    elif ds == 'DEAP' and num_channels == 32:
        ch_names = _DEAP_32_CHANNELS
    else:
        return None

    try:
        import mne
        montage = mne.channels.make_standard_montage('standard_1020')
        # Build case-insensitive position lookup
        pos_raw = montage.get_positions()['ch_pos']
        pos_dict = {k.upper(): v.astype(np.float32) for k, v in pos_raw.items()}
        pos_dict.update({k.upper(): v for k, v in _EXTRA_COORDS.items()})

        coords = []
        missing = []
        for ch in ch_names:
            key = ch.upper()
            if key in pos_dict:
                coords.append(pos_dict[key])
            else:
                missing.append(ch)
                coords.append(np.zeros(3, dtype=np.float32))
        if missing:
            print(f"[build_A_local] Missing electrode positions: {missing} — using origin")
        return np.array(coords, dtype=np.float32)
    except Exception as e:
        print(f"[build_A_local] MNE lookup failed ({e}), falling back to random coords")
        return None


def build_A_local(cfg: dict) -> torch.Tensor:
    """Build fixed local adjacency from real 10-20 electrode 3D distances."""
    C = cfg["data"]["num_channels"]
    delta = cfg["model"].get("delta_local", 0.1)
    dataset = cfg["data"].get("dataset", "SEED")

    coords = _get_electrode_coords(C, dataset)
    if coords is None:
        # Fallback: seeded random (preserves reproducibility, no anatomical meaning)
        np.random.seed(cfg["experiment"]["seed"])
        coords = np.random.randn(C, 3).astype(np.float32)

    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    threshold = np.percentile(dists, delta * 100)
    A = (dists < threshold).astype(np.float32)
    np.fill_diagonal(A, 0)
    return torch.from_numpy(A)


def build_loaders_placeholder(cfg: dict, fold_id: int):
    """Placeholder dataloader factory.

    Replace this with actual LibEER dataloader integration:
        from libeer.data import load_seed_loso
        train_loader, val_loader, test_loader = load_seed_loso(cfg, fold_id)

    Returns (train_loader, val_loader, test_loader, num_train_subjects).
    """
    from torch.utils.data import DataLoader, TensorDataset

    C = cfg["data"]["num_channels"]
    F = cfg["data"]["num_bands"]
    S = cfg["data"].get("sequence_length", 9)
    n_folds = cfg["evaluation"]["n_folds"]
    num_classes = cfg["model"]["num_classes"]
    bs = cfg["training"]["batch_size"]

    num_train_subjects = n_folds - 1
    n_train = 200 * num_train_subjects
    n_val = 50
    n_test = 200

    def make_loader(n, sids_range):
        x = torch.randn(n, S, C, F)
        y = torch.randint(0, num_classes, (n,))
        sid = torch.randint(sids_range[0], sids_range[1], (n,))
        return DataLoader(TensorDataset(x, y, sid), batch_size=bs, shuffle=True)

    train_loader = make_loader(n_train, (0, num_train_subjects))
    val_loader = make_loader(n_val, (0, num_train_subjects))
    test_loader = make_loader(n_test, (0, 1))  # single test subject

    return train_loader, val_loader, test_loader, num_train_subjects


def main():
    parser = argparse.ArgumentParser(description="DG-MAGNet Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--variant", type=str, default=None, help="Ablation variant name")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    cfg = load_config(args.config, args.variant)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: {cfg['experiment']['name']}")
    if args.variant:
        print(f"Variant: {args.variant}")

    results = run_loso(
        cfg,
        build_loaders_fn=build_loaders_seed,
        build_A_local_fn=build_A_local,
        device=device,
    )

    return results


if __name__ == "__main__":
    main()
