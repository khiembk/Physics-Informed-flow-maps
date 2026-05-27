# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This codebase implements **physics-informed flow maps** for solving constrained PDEs, built on the self-distillation framework (LSD, PSD, ESD). The core algorithm learns flow maps X(s,t,x) between states using stochastic interpolants and JAX/Flax.

## Key Commands

### Training Models
```bash
# Training with the checker config (template for PDE problems)
python py/launchers/learn.py \
    --cfg_path configs.checker \
    --slurm_id 0 \
    --dataset_location /path/to/datasets \
    --output_folder /path/to/outputs
```

### Code Formatting
```bash
black py/ --line-length 100
```

## High-Level Architecture

### Core Training Loop (`launchers/learn.py`)
1. **Single-node multi-GPU training** via JAX `pmap` for data parallelism
2. **State management** with EMA (Exponential Moving Average) for stable training
3. **Loss computation** selecting between diagonal/interpolant terms, PSD, LSD, or ESD losses

### Loss Functions (`common/losses.py`)
- **Diagonal/Interpolant**: Basic velocity matching loss
- **PSD** (Progressive Self-Distillation): Two-step distillation with stopgrad options (`uniform`, `midpoint`)
- **LSD** (Lagrangian Self-Distillation): Time derivative matching (`convex`, `none`)
- **ESD** (Eulerian Self-Distillation): Spatial derivative matching (`full`, `convex`, `none`)

### Configuration System (`configs/`)
Configurations use `ml_collections.ConfigDict`:
- `config.training`: Loss types, stopgrad strategies, EMA factors
- `config.problem`: Dataset target, interpolant type, dimensions
- `config.network`: Architecture parameters
- `config.optimization`: Learning rates, schedules, batch sizes
- `config.logging`: WandB settings, visualization frequency

Use `configs/checker.py` as the template when adding a new PDE dataset.

### Adding a New Dataset
1. Add a loader function in `common/datasets.py` inside `setup_target`
2. Create a config in `configs/` (copy from `configs/checker.py`)
3. The dataset should return a flat NumPy array of shape `(n_samples, d)` or use `np_to_tfds`

### Network Architecture (`common/edm2_net.py`, `common/flow_map.py`)
- **EDM2 UNet**: MPConv layers with sphere weight normalization, time-conditioned
- **Flow Map**: Wraps the network to compute X(s,t,x) and potential φ(s,t,x)
- For low-dimensional PDE states, use the MLP network type (`config.network.network_type = "mlp"`)

### State Management (`common/state_utils.py`)
- **EMATrainState**: Flax TrainState with multiple EMA parameter copies
- **StaticArgs**: Immutable config and function references

### Multi-GPU Training (`common/dist_utils.py`)
- Automatic data parallelism via JAX `pmap`
- Single-node only (no multi-node support)

## Algorithm Concepts

### Stochastic Interpolants
- I_t = α(t)x₀ + β(t)x₁ — smooth path between base x₀ and target x₁
- İ_t — time derivative used as velocity target

### Self-Distillation Loss Types
- **LSD**: Matches time derivatives ∂_t X(s,t,x)
- **PSD**: Matches two-step composition X(u,t,X(s,u,x)) ≈ X(s,t,x)
- **ESD**: Matches spatial derivatives ∂_s X(s,t,x)

### Gradient Stopping Strategies
- `none`: Full gradients
- `convex`: Stop gradients on teacher evaluations (most common)
- `full`: Stop all gradients including spatial Jacobian (ESD only)

## Dependencies

- JAX/JAXlib 0.4.26+
- Flax 0.8.2+
- Optax 0.2.2+
- ml_collections 0.1.1+
- TensorFlow 2.15+ (data loading pipeline only)
- WandB 0.16.5+
- Black 24.3.0 (formatting)
