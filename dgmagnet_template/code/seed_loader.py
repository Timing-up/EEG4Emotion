"""
Standalone SEED Dataset Loader (no LibEER dependency)
======================================================
Reads SEED ExtractedFeatures .mat files directly.

SEED labels (per trial, same for all subjects/sessions):
  1=positive, 0=neutral, -1=negative → mapped to 2, 1, 0

File naming: {subject_id}_{date}.mat
Each .mat contains de_LDS1 ... de_LDS15, shape (62, T, 5).
"""
from __future__ import annotations

import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, TensorDataset

# SEED trial emotion labels: 15 trials per session
# -1=negative(0), 0=neutral(1), 1=positive(2)
_SEED_TRIAL_LABELS_RAW = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
_SEED_TRIAL_LABELS = [l + 1 for l in _SEED_TRIAL_LABELS_RAW]  # map to 0,1,2


def _load_subject_session(mat_path: str, seq_len: int):
    """
    Load one subject-session .mat file.
    Returns:
        data:   (N, C, F)  where N = sum of T across 15 trials (floor-divided by 1)
        labels: (N,)
    """
    mat = sio.loadmat(mat_path)
    all_x, all_y = [], []
    for trial_idx in range(15):
        key = f"de_LDS{trial_idx + 1}"
        if key not in mat:
            continue
        feat = mat[key]          # (62, T, 5)
        feat = feat.transpose(1, 0, 2)  # → (T, 62, 5)
        label_val = _SEED_TRIAL_LABELS[trial_idx]
        labels = np.full(len(feat), label_val, dtype=np.int64)
        all_x.append(feat)
        all_y.append(labels)

    if not all_x:
        return None, None
    data = np.concatenate(all_x, axis=0).astype(np.float32)
    labels = np.concatenate(all_y, axis=0).astype(np.int64)
    return data, labels


def _make_sequences(data: np.ndarray, labels: np.ndarray, seq_len: int):
    """Group (N, C, F) into (M, S, C, F) non-overlapping sequences."""
    N = len(data)
    n_seqs = N // seq_len
    if n_seqs == 0:
        pad = seq_len - N
        data = np.concatenate([data, data[:pad]], axis=0)
        labels = np.concatenate([labels, labels[:pad]], axis=0)
        n_seqs = 1
    data = data[:n_seqs * seq_len]
    labels = labels[:n_seqs * seq_len]
    seq_data = data.reshape(n_seqs, seq_len, *data.shape[1:])
    seq_labels = labels.reshape(n_seqs, seq_len)
    seq_labels = np.array([np.bincount(row, minlength=3).argmax() for row in seq_labels])
    return seq_data, seq_labels


def _collect_subject_data(mat_paths: list[str], seq_len: int):
    """Load and concatenate all sessions for each subject."""
    # Group by subject id (parsed from filename)
    from collections import defaultdict
    subj_files: dict[int, list[str]] = defaultdict(list)
    for p in mat_paths:
        fname = os.path.basename(p)
        part = fname.split("_")[0]
        if not part.isdigit():
            continue  # skip label.mat and other non-subject files
        subj_id = int(part) - 1  # 0-indexed
        subj_files[subj_id].append(p)

    subj_data = {}
    subj_labels = {}
    for sid in sorted(subj_files.keys()):
        xs, ys = [], []
        for path in sorted(subj_files[sid]):
            x, y = _load_subject_session(path, seq_len)
            if x is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            subj_data[sid] = np.concatenate(xs, axis=0)
            subj_labels[sid] = np.concatenate(ys, axis=0)
    return subj_data, subj_labels


def build_loaders_seed(cfg: dict, fold_id: int):
    """
    Build train/val/test DataLoaders for one SEED LOSO fold.

    Args:
        cfg:     merged DG-MAGNet config dict
        fold_id: 0-based test subject index (0..14)

    Returns:
        (train_loader, val_loader, test_loader, num_train_subjects)
    """
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    seq_len = dcfg.get("sequence_length", 9)
    val_frac = cfg["evaluation"].get("val_split_from_train", 0.1)
    bs = tcfg["batch_size"]

    data_root = os.path.expanduser(dcfg["root"])
    feat_dir = os.path.join(data_root, "ExtractedFeatures")
    mat_paths = sorted(glob.glob(os.path.join(feat_dir, "*.mat")))
    assert len(mat_paths) > 0, f"No .mat files found in {feat_dir}"

    subj_data, subj_labels = _collect_subject_data(mat_paths, seq_len)
    num_subjects = len(subj_data)
    assert num_subjects == 15, f"Expected 15 subjects, got {num_subjects}"

    test_idx = fold_id % num_subjects
    train_indices = [i for i in range(num_subjects) if i != test_idx]

    # Val split BEFORE recording num_train_subjects so model is built
    # with exactly the number of subjects that appear in training batches.
    n_val_subs = max(1, int(round(len(train_indices) * val_frac)))
    val_indices = train_indices[-n_val_subs:]
    train_indices = train_indices[:-n_val_subs]
    num_train_subjects = len(train_indices)  # 13 for SEED (14-1 val)

    def collect(indices, assign_subject=True):
        all_x, all_y, all_sid = [], [], []
        for new_sid, orig_sid in enumerate(indices):
            d = subj_data[orig_sid]
            l = subj_labels[orig_sid]
            seq_x, seq_y = _make_sequences(d, l, seq_len)
            all_x.append(seq_x)
            all_y.append(seq_y)
            sid_val = new_sid if assign_subject else 0
            all_sid.append(np.full(len(seq_x), sid_val, dtype=np.int64))
        x = np.concatenate(all_x, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.int64)
        sid = np.concatenate(all_sid, axis=0).astype(np.int64)
        return x, y, sid

    train_x, train_y, train_sid = collect(train_indices, assign_subject=True)
    val_x, val_y, val_sid = collect(val_indices, assign_subject=False)
    test_x, test_y, _ = collect([test_idx], assign_subject=False)

    # Subject z-score normalization from training data
    mean = train_x.mean(axis=(0, 1), keepdims=True)   # (1,1,C,F)
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
                          num_workers=0, pin_memory=False)

    train_loader = make_loader(train_x, train_y, train_sid, shuffle=True)
    val_loader = make_loader(val_x, val_y, val_sid, shuffle=False)
    test_loader = make_loader(test_x, test_y,
                              np.zeros(len(test_x), dtype=np.int64), shuffle=False)

    print(f"  Fold {fold_id}: train={len(train_x)} val={len(val_x)} test={len(test_x)} "
          f"| shape={train_x.shape[1:]} | train_subs={num_train_subjects}")

    return train_loader, val_loader, test_loader, num_train_subjects
