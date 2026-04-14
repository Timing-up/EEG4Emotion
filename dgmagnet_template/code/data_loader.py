"""
LibEER Dataloader Integration for DG-MAGNet
============================================

LibEER outputs per-sample features of shape (C, F) — single time windows.
DG-MAGNet expects sequences (B, S, C, F).

This module:
1. Calls LibEER's get_data / merge_to_part / get_split_index / index_to_data
2. Groups consecutive windows into sequences of length S
3. Assigns subject IDs (0-indexed within training fold)
4. Returns PyTorch DataLoaders with batch shape (x, labels, subject_ids)
   where x: (B, S, C, F), labels: (B,), subject_ids: (B,)
"""
from __future__ import annotations

import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _import_libeer(libeer_root: str = None):
    """Add LibEER to sys.path so its modules can be imported."""
    if libeer_root is None:
        libeer_root = os.path.expanduser("~/LibEER/LibEER")
    if libeer_root not in sys.path:
        sys.path.insert(0, libeer_root)


def _make_sequences(data: np.ndarray, labels: np.ndarray, seq_len: int):
    """
    Group consecutive samples into non-overlapping sequences.

    Args:
        data:   (N, C, F)
        labels: (N,) int
        seq_len: S

    Returns:
        seq_data:   (M, S, C, F)
        seq_labels: (M,) — majority vote label per sequence
    """
    N = len(data)
    n_seqs = N // seq_len
    if n_seqs == 0:
        # Pad if not enough samples
        pad = seq_len - N
        data = np.concatenate([data, data[:pad]], axis=0)
        labels = np.concatenate([labels, labels[:pad]], axis=0)
        n_seqs = 1

    data = data[:n_seqs * seq_len]
    labels = labels[:n_seqs * seq_len]

    seq_data = data.reshape(n_seqs, seq_len, *data.shape[1:])      # (M, S, C, F)
    seq_labels = labels.reshape(n_seqs, seq_len)
    # Majority vote per sequence
    seq_labels = np.array([np.bincount(row).argmax() for row in seq_labels])
    return seq_data, seq_labels


def build_loaders_libeer(cfg: dict, fold_id: int, libeer_root: str = None):
    """
    Build train/val/test DataLoaders for one LOSO fold using LibEER.

    Args:
        cfg: merged DG-MAGNet config dict
        fold_id: index of the test subject (0-based)
        libeer_root: path to LibEER/LibEER directory

    Returns:
        (train_loader, val_loader, test_loader, num_train_subjects)
    """
    _import_libeer(libeer_root)

    from data_utils.load_data import get_data
    from data_utils.split import merge_to_part, index_to_data, get_split_index

    dcfg = cfg["data"]
    tcfg = cfg["training"]
    seq_len = dcfg.get("sequence_length", 9)
    val_frac = cfg["evaluation"].get("val_split_from_train", 0.1)
    bs = tcfg["batch_size"]

    # Build a minimal LibEER-compatible setting object
    setting = _build_setting(cfg)

    # Load and preprocess data
    data, label, channels, feature_dim, num_classes = get_data(setting)

    # Merge into (1, num_subjects, n_samples, C, F) for subject-independent
    data, label = merge_to_part(data, label, setting)
    # data[0] is list of per-subject arrays, each (n_i, C, F)
    subject_data = data[0]    # list of length num_subjects
    subject_label = label[0]  # list of length num_subjects

    num_subjects = len(subject_data)
    # fold_id is the test subject index
    test_idx = fold_id % num_subjects
    train_indices = [i for i in range(num_subjects) if i != test_idx]
    num_train_subjects = len(train_indices)

    # Split training into train/val by held-out subjects
    n_val_subs = max(1, int(round(num_train_subjects * val_frac)))
    val_indices = train_indices[-n_val_subs:]
    train_indices = train_indices[:-n_val_subs]

    def collect(indices, assign_subject=True):
        all_x, all_y, all_sid = [], [], []
        for new_sid, orig_sid in enumerate(indices):
            d = np.array(subject_data[orig_sid])    # (N, C, F) or (N, C*F)
            l = np.array(subject_label[orig_sid])   # (N,) or (N, num_classes)

            # Handle one-hot labels
            if l.ndim == 2:
                l = l.argmax(axis=-1)
            l = l.astype(np.int64)

            # Reshape if needed: some LibEER outputs (N, C*F) flat
            if d.ndim == 2:
                d = d.reshape(len(d), channels, feature_dim)

            seq_x, seq_y = _make_sequences(d, l, seq_len)
            all_x.append(seq_x)
            all_y.append(seq_y)
            if assign_subject:
                all_sid.append(np.full(len(seq_x), new_sid, dtype=np.int64))
            else:
                all_sid.append(np.zeros(len(seq_x), dtype=np.int64))  # test sub gets id 0

        x = np.concatenate(all_x, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.int64)
        sid = np.concatenate(all_sid, axis=0).astype(np.int64)
        return x, y, sid

    train_x, train_y, train_sid = collect(train_indices, assign_subject=True)
    val_x, val_y, val_sid = collect(val_indices, assign_subject=False)
    test_x, test_y, test_sid = collect([test_idx], assign_subject=False)

    # Normalize: subject z-score from training data
    mean = train_x.mean(axis=(0, 1), keepdims=True)  # (1, 1, C, F)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    def make_loader(x, y, sid, shuffle):
        ds = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(sid),
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle,
                          num_workers=2, pin_memory=True)

    train_loader = make_loader(train_x, train_y, train_sid, shuffle=True)
    val_loader = make_loader(val_x, val_y, val_sid, shuffle=False)
    test_loader = make_loader(test_x, test_y, test_sid, shuffle=False)

    print(f"  Fold {fold_id}: train={len(train_x)} val={len(val_x)} test={len(test_x)} "
          f"| shape={train_x.shape[1:]} | train_subs={num_train_subjects}")

    return train_loader, val_loader, test_loader, num_train_subjects


def _build_setting(cfg: dict):
    """Build a minimal LibEER Setting object from DG-MAGNet config."""
    _import_libeer()
    from config.setting import Setting

    dcfg = cfg["data"]
    dataset = dcfg["dataset"].lower()

    # Map dataset name to LibEER dataset string
    dataset_map = {
        "seed": "seed_de_lds",
        "seed-iv": "seediv_de_lds",
        "deap": "deap",
    }
    libeer_dataset = dataset_map.get(dataset, dataset)

    dataset_path = os.path.expanduser(dcfg["root"])

    # Band config: LibEER DE uses 5 standard bands
    extract_bands = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]

    return Setting(
        dataset=libeer_dataset,
        dataset_path=dataset_path,
        pass_band=[0.3, 50.0],
        extract_bands=extract_bands,
        time_window=dcfg.get("window_seconds", 1),
        overlap=dcfg.get("overlap", 0.0),
        sample_length=1,
        stride=1,
        seed=cfg["experiment"]["seed"],
        feature_type="de_lds",
        only_seg=False,
        experiment_mode="subject-independent",
        normalize=False,     # we do our own z-score normalization
        split_type="leave-one-out",
        sessions=[1, 2, 3],  # use all 3 SEED sessions
        onehot=False,
        label_used=None,
        bounds=None,
    )
