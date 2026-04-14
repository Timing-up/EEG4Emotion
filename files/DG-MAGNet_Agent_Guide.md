# DG-MAGNet: Agent Execution Guide

> **Mission**: Build DG-MAGNet — a cross-subject EEG emotion recognition framework combining Invariant-Specific Graph Decomposition (ISGD), adversarial domain generalization, and Test-Time Graph Adaptation (TTGA) — targeting AAAI / IJCAI submission.
>
> **Target metrics**: SEED LOSO ≥ 89%, SEED-IV ≥ 75%, DEAP ≥ 70%
>
> **Philosophy**: Reuse open-source maximally. Write ~300 lines of original code, integrate thousands.

---

## 0. Ground Rules for the Agent

1. **Never rewrite what exists**. Before implementing anything, search the referenced repos for an existing implementation.
2. **Fail fast, fail loud**. After every milestone, run a smoke test. If the baseline number is >3% off expected, stop and report — don't paper over it.
3. **Log everything**. Use `wandb` or `tensorboard` from day 1. Every experiment gets a unique run ID.
4. **Version control**. Git commit after every working milestone. Branch per experiment.
5. **Report progress** to the human at every checkpoint marked `🛑 CHECKPOINT`.
6. **Ask for clarification** when facing architectural ambiguity. Do NOT guess on math.

---

## 1. Environment Setup

### 1.1 Hardware requirements
- Linux (Ubuntu 20.04+)
- NVIDIA GPU with CUDA 11.8+ (Mamba does not support Windows/AMD)
- ≥ 24 GB GPU memory recommended (A100 / 4090 / 3090)
- ≥ 100 GB disk for datasets

### 1.2 Install commands

```bash
conda create -n dgmagnet python=3.10 -y
conda activate dgmagnet

# PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Mamba (critical: install in this order)
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1

# Graph
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Standard
pip install numpy scipy scikit-learn pandas matplotlib seaborn
pip install wandb tensorboard tqdm einops
pip install mne  # EEG preprocessing
```

### 1.3 Clone dependencies

```bash
mkdir -p ~/dgmagnet && cd ~/dgmagnet

git clone https://github.com/XJTU-EEG/LibEER.git
git clone https://github.com/DequanWang/tent.git
git clone https://github.com/Linear95/CLUB.git
git clone https://github.com/thuml/Transfer-Learning-Library.git TLL
```

### 1.4 Validation smoke test

```bash
cd LibEER
python -c "import libeer; print('LibEER OK')"
python -c "from mamba_ssm import Mamba; print('Mamba OK')"
python -c "import torch_geometric; print('PyG OK')"
```

🛑 **CHECKPOINT 1**: All imports succeed. Report GPU name and CUDA version.

---

## 2. Dataset Preparation

### 2.1 Datasets to obtain
| Dataset | Source | How to get |
|---|---|---|
| SEED | BCMI @ SJTU | Apply at https://bcmi.sjtu.edu.cn/~seed/seed.html (1–3 days) |
| SEED-IV | BCMI @ SJTU | Same application |
| DEAP | Queen Mary | Apply at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ |

### 2.2 Directory structure

```
~/dgmagnet/
├── data/
│   ├── SEED/
│   │   ├── Preprocessed_EEG/    # .mat files, 15 subjects × 3 sessions
│   │   └── ExtractedFeatures/
│   ├── SEED-IV/
│   └── DEAP/
├── LibEER/
├── code/                          # YOUR code goes here
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── figures/
```

### 2.3 Preprocessing via LibEER
- Use LibEER's `Setting` class with:
  - **Feature**: DE (Differential Entropy), 5 frequency bands (δ, θ, α, β, γ)
  - **Window**: 1 second, no overlap
  - **Filter**: 0.3–50 Hz bandpass
  - **Normalization**: per-subject z-score on training set only

🛑 **CHECKPOINT 2**: Load SEED via LibEER, print tensor shape. Expected: `(N, 62, 5)` for feature format or `(N, T, 62, 5)` for sequence format. Report actual shape.

---

## 3. Codebase Architecture

### 3.1 Directory layout to create

```
code/
├── configs/
│   ├── base.yaml              # defaults
│   ├── seed_loso.yaml
│   └── ablation_*.yaml
├── models/
│   ├── __init__.py
│   ├── dg_magnet.py           # main model
│   ├── isgd.py                # Invariant-Specific Graph Decomposition
│   ├── temporal.py            # Mamba + Attention hybrid
│   └── losses.py              # adv + MI + CE combined
├── adaptation/
│   └── ttga.py                # test-time graph adaptation
├── trainers/
│   ├── base_trainer.py        # LOSO loop
│   └── dg_trainer.py          # with adversarial loss
├── utils/
│   ├── metrics.py
│   ├── visualize.py           # t-SNE, brain topo, adjacency heatmap
│   └── seed.py
├── scripts/
│   ├── train_baseline.py
│   ├── train_dgmagnet.py
│   ├── run_ablation.sh
│   └── run_full_benchmark.sh
└── README.md
```

### 3.2 Model hyperparameters (starting point)

```yaml
# configs/base.yaml
model:
  num_channels: 62          # SEED electrodes (DEAP: 32)
  num_bands: 5              # frequency bands
  d_g: 64                   # graph embedding dim
  d_state: 16               # Mamba state size
  mamba_depth: 4
  attention_heads: 8
  K_cheb: 4                 # Chebyshev order
  L1: 2                     # ChebyNet branch 1 depth
  L2: 2                     # ChebyNet branch 2 depth
  delta: 0.1                # local adjacency threshold (normalized 3D distance)
  dropout: 0.3

isgd:
  low_rank_r: 8             # rank of specific adjacency modulation
  subject_embed_dim: 32

losses:
  alpha_adv: 0.1            # adversarial weight
  alpha_mi: 0.01            # CLUB MI weight
  alpha_reg: 0.001          # Frobenius regularization

training:
  optimizer: AdamW
  lr: 5e-4
  weight_decay: 1e-4
  batch_size: 64
  epochs: 100
  early_stop_patience: 20
  scheduler: cosine

ttga:
  adapt_samples: 5          # unlabeled target samples
  adapt_steps: 5
  adapt_lr: 1e-3
  entropy_weight: 1.0
  frobenius_weight: 0.1
```

---

## 4. Implementation Tasks (Ordered)

### TASK 1: Reproduce baseline (vanilla MAGNet in LibEER)

**Goal**: Confirm LibEER pipeline works, get a ~80% baseline on SEED LOSO.

Steps:
1. Fork LibEER. Add `LibEER/libeer/models/magnet.py`.
2. Implement vanilla MAGNet (no DG, no TTGA) using:
   - `torch_geometric.nn.ChebConv` for ChebyNet branches
   - `mamba_ssm.Mamba` for Mamba block
   - `torch.nn.MultiheadAttention` for attention
3. Register in `LibEER/libeer/models/__init__.py`.
4. Create `MAGNet_train.py` by copying `DGCNN_train.py` and swapping model.
5. Run SEED LOSO with cross_subject setting.

**Expected outcome**: 78–82% accuracy. If lower, debug; if higher, suspect data leakage.

🛑 **CHECKPOINT 3**: Report SEED LOSO accuracy, std, and training time per fold.

---

### TASK 2: Implement ISGD (Invariant-Specific Graph Decomposition)

**File**: `code/models/isgd.py`

Implements:
```
A^k = A_inv^k + lambda * A_spec^k(s)
A_spec^k(s) = U_s @ V_s.T    # low-rank, rank r=8
U_s, V_s generated from subject embedding via MLP
```

Key points:
- `A_inv` is a learnable `nn.Parameter` of shape `(num_channels, num_channels)`, shared across all subjects
- Subject embedding table `nn.Embedding(num_subjects, subject_embed_dim)`
- MLP projects subject embedding to `U_s` and `V_s` of shape `(num_channels, r)`
- Symmetrize: `A = 0.5 * (A + A.T)`, then `ReLU`
- **At inference time on unseen subjects**, initialize a new subject embedding (start from mean of training embeddings)

🛑 **CHECKPOINT 4**: Unit test — feed random input, verify output shapes and grad flow.

---

### TASK 3: Adversarial subject discriminator + CLUB MI loss

**File**: `code/models/losses.py`

Components:
1. **Gradient Reversal Layer** — copy from `TLL/tllib/modules/grl.py`
2. **Subject Discriminator** — 3-layer MLP, output = num_train_subjects logits
3. **Adversarial loss**: cross entropy on discriminator, but feature extractor gets reversed gradient
4. **CLUB MI loss** — copy from `CLUB/mi_estimators.py`, use `CLUBSample` class
5. **Frobenius regularization** on `A_spec`

Final loss:
```
L = L_CE + alpha_adv * L_adv + alpha_mi * L_MI + alpha_reg * ||A_spec||_F^2
```

🛑 **CHECKPOINT 5**: On SEED LOSO, DG-MAGNet (ISGD + adv + MI, no TTGA yet) should improve over vanilla MAGNet by ≥ 3%. Target: 83–86%.

---

### TASK 4: Test-Time Graph Adaptation (TTGA)

**File**: `code/adaptation/ttga.py`

Base it on TENT (`tent/tent.py`, ~300 lines). Modifications:

1. Freeze all parameters except the **new subject's** `A_spec` and subject embedding
2. Collect 5 unlabeled samples from target subject
3. Forward pass → compute entropy loss + Frobenius penalty
4. Backward and step for 5 iterations
5. Then do final prediction on target

Pseudocode:
```python
def ttga_adapt(model, target_samples, steps=5, lr=1e-3):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze target subject's specific adjacency + embedding
    target_embed = nn.Parameter(init_from_mean_embedding())
    optimizer = torch.optim.Adam([target_embed], lr=lr)

    for _ in range(steps):
        logits = model(target_samples, subject_embed=target_embed)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.log()).sum(-1).mean()
        reg = target_embed.norm(p=2) ** 2
        loss = entropy + 0.1 * reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return target_embed
```

🛑 **CHECKPOINT 6**: With TTGA, SEED LOSO target: **89% +**. Report gain from TTGA alone.

---

## 5. Experiments to Run

### Exp 1: Main comparison (MUST)

| Dataset | Protocol | Baselines |
|---|---|---|
| SEED | LOSO 15-fold | DGCNN, RGNN, EmT, PR-PL, MS-MDA, SS-EMERGE*, vanilla MAGNet, **DG-MAGNet** |
| SEED-IV | LOSO 15-fold | same |
| DEAP | LOSO 32-fold | same (valence & arousal binary) |

Report: mean ± std accuracy, macro-F1, paired Wilcoxon test vs DG-MAGNet.

*If SS-EMERGE code not available, cite paper numbers.

### Exp 2: Ablation study (MUST)

| Variant | ISGD | Adv | MI | TTGA | Expected SEED ACC |
|---|---|---|---|---|---|
| Vanilla MAGNet | ❌ | ❌ | ❌ | ❌ | ~80% |
| + ISGD | ✅ | ❌ | ❌ | ❌ | ~82% |
| + ISGD + Adv | ✅ | ✅ | ❌ | ❌ | ~84% |
| + ISGD + Adv + MI | ✅ | ✅ | ✅ | ❌ | ~85% |
| **Full (DG-MAGNet)** | ✅ | ✅ | ✅ | ✅ | **~89%** |

### Exp 3: TTGA sensitivity

Sweep: `adapt_samples ∈ {1,3,5,10,20}` × `adapt_steps ∈ {1,3,5,10}`. Produce heatmap.

### Exp 4: Invariance analysis

1. t-SNE of `H_inv` and `H_spec`: invariant should mix subjects, specific should cluster by subject
2. Linear probe: predict subject ID from `H_inv` → should be near chance

### Exp 5: Interpretability

1. Visualize learned `A_inv` on 2D brain topography (use MNE's `plot_connectivity_circle`)
2. Per-band importance: Grad-CAM on frequency input
3. Compare learned connections to known emotion-related regions (PFC, temporal lobes)

---

## 6. Evaluation Protocol (Strict)

1. **LOSO**: for each of N subjects, train on N-1, test on 1. Repeat for all subjects.
2. **Per-fold**: subject-wise z-score normalization using training subjects' statistics ONLY
3. **Random seed**: run each experiment with 3 seeds {2024, 2025, 2026}, report mean
4. **Early stopping**: on a held-out 10% of training subjects (NOT the test subject)
5. **Reporting**: mean ± std across subjects × seeds
6. **Significance**: Wilcoxon signed-rank test vs best baseline, Bonferroni corrected

---

## 7. Writing Phase

Only start writing AFTER all experiments converge.

### Section order
1. **Abstract** — problem → insight → method → top number
2. **Intro** — 4 paragraphs: problem, observation, insight, contributions
3. **Related Work** — 3 subsections: GNN for EEG, Cross-subject DG, Test-time adaptation
4. **Method** — ISGD, adversarial disentanglement, TTGA (with loss equations)
5. **Experiments** — setup, main results (Exp 1), ablations (Exp 2), analyses (Exp 3-5)
6. **Discussion / Limitations** — 1 paragraph acknowledging session-shift not covered
7. **Conclusion**

### Figures needed
- Fig 1: Architecture diagram (redraw, fix the spelling errors from original MAGNet fig)
- Fig 2: Motivating analysis — different subjects' learned adjacency matrices
- Fig 3: Main results bar chart
- Fig 4: Ablation table
- Fig 5: TTGA heatmap
- Fig 6: t-SNE of invariant vs specific features
- Fig 7: Brain topography of learned A_inv

---

## 8. Deliverables Checklist

- [ ] Working code repo with README
- [ ] Trained checkpoints for 3 datasets
- [ ] `results.csv` with all numbers
- [ ] All figures in `outputs/figures/` (PDF vector format)
- [ ] Paper draft (LaTeX)
- [ ] Supplementary material (hyperparams, extra ablations)
- [ ] Code cleanup + public GitHub release (anonymized for submission)

---

## 9. Risk Register

| Risk | Mitigation |
|---|---|
| Mamba install fails on cluster | Fallback to `torch.nn` SSM reimpl or use Linear Attention |
| SEED dataset approval delayed | Start with DEAP first (public) |
| Vanilla MAGNet can't reach 80% | Increase `d_g`, try 7 bands instead of 5, check normalization |
| DG-MAGNet gains < 3% | Run Exp 4 early to diagnose — if `H_inv` still encodes subject, adv weight too low |
| TTGA causes catastrophic forgetting | Lower `adapt_lr`, increase Frobenius weight |
| GPU OOM on SEED | Reduce batch to 32, use gradient accumulation |
| Results variance too high | Increase seeds from 3 to 5, check normalization bug |

---

## 10. Communication Protocol

At each 🛑 CHECKPOINT, report to human:
1. What was done
2. Metric numbers (if applicable)
3. Anything surprising or broken
4. Proposed next step

Do not proceed past a CHECKPOINT without either (a) hitting the expected metric or (b) explicit human approval to continue with degraded numbers.

---

## Appendix A: Key References

- **LibEER**: https://github.com/XJTU-EEG/LibEER
- **Mamba**: https://github.com/state-spaces/mamba
- **TENT** (test-time adaptation): https://github.com/DequanWang/tent
- **CLUB** (MI estimation): https://github.com/Linear95/CLUB
- **Transfer-Learning-Library** (DANN etc): https://github.com/thuml/Transfer-Learning-Library
- **EmT** (GCN + Transformer EEG): search `ding emt eeg github`
- **PR-PL** (EEG cross-subject SOTA): search `pr-pl eeg github`

## Appendix B: Citation Targets (read these first)

1. Gu & Dao 2023 — Mamba
2. Wang et al. ICLR 2021 — TENT
3. Ganin et al. JMLR 2016 — DANN
4. Cheng et al. ICML 2020 — CLUB
5. Zhou et al. TAFFC 2023 — PR-PL
6. Defferrard et al. NeurIPS 2016 — ChebyNet
7. Song et al. TAFFC 2020 — DGCNN (EEG baseline)
8. Zhong et al. TAFFC 2020 — RGNN

---

**End of guide. Agent: begin at Section 1.**
