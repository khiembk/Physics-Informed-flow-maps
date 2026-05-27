# Physics-Informed Flow Maps

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX 0.4.26+](https://img.shields.io/badge/JAX-0.4.26+-green.svg)](https://github.com/google/jax)

Learning flow maps X(s,t,x) to solve constrained PDEs via self-distillation.

## Background

This project applies the **flow map self-distillation** framework to physics-informed learning. A flow map X(s,t,x) transports an initial state x at time s to the state at time t, satisfying the tangent condition:

```
∂_t X(s,t,x) = b(t, X(s,t,x))
```

where b is the underlying velocity field. Instead of integrating ODEs at inference time, the flow map is learned directly via self-distillation, enabling single or few-step rollouts.

Three training algorithms are available:

- **LSD** (Lagrangian Self-Distillation) — matches ∂_t X(s,t,x) to the velocity field; most stable in practice
- **PSD** (Progressive Self-Distillation) — bootstraps large steps from small steps via composition
- **ESD** (Eulerian Self-Distillation) — minimizes the PDE residual of the tangent condition

## Installation

### Requirements
- Python 3.9+
- CUDA 11.8+ or 12.0+

### Setup

**1. Create environment**
```bash
conda create -n flowmaps python=3.9
conda activate flowmaps
```

**2. Install JAX** for your CUDA version:
```bash
# CUDA 12.x
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11.8
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CPU only
pip install --upgrade jax
```

**3. Install dependencies**
```bash
pip install \
    flax==0.8.2 \
    optax==0.2.2 \
    ml_collections==0.1.1 \
    tensorflow==2.15.0 \
    wandb==0.16.5 \
    matplotlib==3.7.0 \
    seaborn==0.12.2 \
    scipy==1.10.1 \
    tqdm==4.65.0
```

**4. Verify**
```bash
python -c "import jax; print(f'JAX {jax.__version__} | Devices: {jax.devices()}')"
```


## Quick start

### Training

```bash
python py/launchers/learn.py \
    --cfg_path configs.checker \
    --slurm_id 0 \
    --output_folder /path/to/outputs
```

Select the algorithm via `slurm_id`:

| ID | Algorithm |
|----|-----------|
| 0  | LSD       |
| 1  | PSD-uniform |
| 2  | PSD-midpoint |
| 3  | ESD       |


## Adding a new PDE dataset

1. Add a loader in `py/common/datasets.py` inside `setup_target` — return `(cfg, ds, prng_key)` where `ds` is a batched iterator yielding NumPy arrays of shape `(batch, d)`.
2. Copy `py/configs/checker.py` and set `config.problem.target`, `config.problem.d`, and network/optimization hyperparameters.
3. Train with `--cfg_path configs.<your_config>`.


## Multi-GPU training

JAX automatically uses all visible GPUs on a single node:

```bash
# All GPUs
python py/launchers/learn.py --cfg_path configs.checker --slurm_id 0

# Restrict to specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python py/launchers/learn.py --cfg_path configs.checker --slurm_id 0
```


## SLURM cluster

```bash
sbatch slurm_scripts/checker.sbatch
```

Edit the script to set your account, partition, module loads, and paths before submitting.


## Weights & Biases

```bash
wandb login
export WANDB_ENTITY="your-username"

# Disable if needed
export WANDB_MODE=offline
```

Metrics logged per step: loss, gradient norm, learning rate, step time, and sample visualizations every `visual_freq` steps.


## Project structure

```
├── py/
│   ├── configs/
│   │   └── checker.py           # 2D checkerboard (template for PDE configs)
│   ├── common/
│   │   ├── losses.py            # LSD, PSD, ESD loss implementations
│   │   ├── flow_map.py          # Flow map network wrapper
│   │   ├── edm2_net.py          # EDM2 UNet architecture
│   │   ├── interpolant.py       # Stochastic interpolants
│   │   ├── datasets.py          # Dataset loading (add PDE datasets here)
│   │   ├── state_utils.py       # EMA training state
│   │   ├── dist_utils.py        # Multi-GPU utilities
│   │   ├── loss_args.py         # Loss argument sampling
│   │   ├── logging.py           # WandB logging and visualization
│   │   ├── network_utils.py     # Network initialization
│   │   └── updates.py           # Optimizer and LR schedules
│   └── launchers/
│       └── learn.py             # Main training script
├── notebooks/
│   └── checker.ipynb            # 2D checkerboard visualization
└── slurm_scripts/
    └── checker.sbatch
```


## Based on

This codebase builds on the flow map self-distillation framework from:

```bibtex
@misc{boffi2025buildconsistencymodellearning,
      title={How to build a consistency model: Learning flow maps via self-distillation},
      author={Nicholas M. Boffi and Michael S. Albergo and Eric Vanden-Eijnden},
      year={2025},
      eprint={2505.18825},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.18825},
}
```


## License

MIT License.
