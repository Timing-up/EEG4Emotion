# DG-MAGNet Project Template

Starter scaffold for Cross-Subject EEG Emotion Recognition with ISGD + DG + TTGA.

## Structure
```
configs/              # YAML configs (base + per-dataset + ablation)
code/models/          # ISGD, temporal, DG-MAGNet main model, losses
code/adaptation/      # TTGA test-time adaptation
code/trainers/        # [TODO] LOSO trainer
code/utils/           # [TODO] metrics, visualization, seed
code/scripts/         # [TODO] train/ablation bash scripts
```

## What's provided (ready to use)
- `configs/*.yaml` — complete hyperparameter configs
- `models/isgd.py` — Invariant-Specific Graph Decomposition ✓ unit test included
- `models/temporal.py` — Mamba-Attention hybrid ✓ with GRU fallback
- `models/dg_magnet.py` — Full model integrating spatial + temporal + classifier
- `models/losses.py` — CE + GRL Adversarial + CLUB MI + Frobenius reg
- `adaptation/ttga.py` — Test-time graph adaptation

## What's TODO (Agent must implement)
- `trainers/dg_trainer.py` — LOSO training loop with all losses
- `utils/metrics.py` — accuracy, macro-F1, Wilcoxon tests
- `utils/visualize.py` — t-SNE, brain topo, ablation bars
- `scripts/train_*.sh` — runner scripts
- Integration with LibEER dataloader (preprocessing stays in LibEER)

## Testing the skeletons
```bash
cd code
python -m models.isgd        # runs unit test
python -m models.temporal    # shape check
```

## Quick start for Agent
1. Clone LibEER and install mamba-ssm (see main guide)
2. Copy this template to your project root
3. Implement trainer by adapting LibEER's `DGCNN_train.py`
4. Wire dataloader output -> model forward signature: (x, subject_ids)
5. Run vanilla ablation first to confirm baseline ~80%
6. Progressively enable ISGD, Adv, MI, TTGA

See `DG-MAGNet_Agent_Guide.md` for the full execution protocol.
