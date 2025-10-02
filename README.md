# Flow Maps via Self-Distillation: 10-100× Faster Generation in One Step

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.18825-b31b1b.svg)](https://arxiv.org/abs/2505.18825)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-green.svg)](https://github.com/google/jax)

Official implementation of **"How to build a consistency model: Learning flow maps via self-distillation"** (NeurIPS 2025).

**Generate high-quality samples in a single step** — no iterative denoising, no pre-trained teachers, just direct generation.

![Overview](figs/overview.png)

## 🚀 Why Flow Maps?

Traditional diffusion models produce stunning results but require 10-1000 sequential steps for generation. **Flow maps change the game** by learning to jump directly from noise to data in one step, achieving:

- **10-100× faster inference** than standard flow models
- **One-step generation** without quality degradation
- **No pre-trained teacher models** required
- **Mathematically grounded** training via the tangent condition

## 🎯 Key Innovation: The Tangent Condition

We introduce three self-distillation algorithms that train flow maps from scratch by exploiting a fundamental mathematical relationship — the **tangent condition** — that connects the velocity field to derivatives of the flow map:

1. **LSD (Lagrangian Self-Distillation)** ⭐ **Best Performance**
   - Matches time derivatives without spatial gradients
   - Avoids self-consistent bootstrapping issues
   - Provable 2-Wasserstein error bounds

2. **PSD (Progressive Self-Distillation)**
   - Two-step teacher distillation with composition
   - Uniform and midpoint variants

3. **ESD (Eulerian Self-Distillation)**
   - Matches spatial derivatives of the flow map
   - Direct enforcement of the tangent condition

**Key Result**: LSD consistently outperforms all other methods across CIFAR-10, CelebA-64, and AFHQ-64.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flow-maps.git
cd flow-maps

# Create conda environment
conda create -n flowmaps python=3.9
conda activate flowmaps

# Install JAX (GPU version - adjust for your CUDA version)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install flax==0.8.2 optax==0.2.2 ml_collections==0.1.1 tensorflow==2.15.0 wandb==0.16.5 black==24.3.0
```

## 🏃 Quick Start

### Training a Flow Map

Train your own one-step generative model:

```bash
# Train LSD on CIFAR-10 (our best method)
python py/launchers/learn.py \
    --cfg_path configs.cifar10 \
    --slurm_id 0 \
    --dataset_location /path/to/datasets \
    --output_folder /path/to/outputs

# Train on other datasets
python py/launchers/learn.py --cfg_path configs.celeba64 --slurm_id 0  # CelebA-64
python py/launchers/learn.py --cfg_path configs.afhq64 --slurm_id 0    # AFHQ-64
python py/launchers/learn.py --cfg_path configs.checker --slurm_id 0   # 2D Checker
```

**Experiment variants** (controlled by `slurm_id`):
- `0`: LSD with convex stopgrad ⭐ **Recommended**
- `1`: PSD uniform with convex stopgrad
- `2`: PSD midpoint with convex stopgrad
- `3`: ESD with full stopgrad

### Generating Samples and Computing FID

Once trained, evaluate your models:

```bash
# Sample from a trained checkpoint
python py/launchers/sample_model.py \
    --config_path /path/to/config.py \
    --checkpoint_path /path/to/checkpoint.pkl \
    --n_samples 10000 \
    --output_path samples.npy

# Compute FID scores
python py/launchers/sample_and_calc_fid.py \
    --cfg_path configs.cifar10 \
    --slurm_id 0 \
    --checkpoint /path/to/checkpoint.pkl \
    --n_steps 1 \
    --n_images 50000
```

**One-step generation!** Set `--n_steps 1` to generate samples in a single forward pass.

## 🖥️ Multi-GPU Training

The codebase supports efficient single-node multi-GPU training via JAX's `pmap`:

```bash
# Automatically uses all visible GPUs
python py/launchers/learn.py --cfg_path configs.cifar10 --slurm_id 0
```

Batch sizes are automatically sharded across available devices. For cluster deployment, example SLURM scripts are provided in `slurm_scripts/` (adjust paths for your system).

## 📊 Reproducibility

All experiments use EDM2-style UNet architectures with:
- Positional embeddings for time conditioning
- Square root learning rate schedules with warmup
- Diagonal fraction of 0.75 for flow matching
- Consistent FID evaluation protocols

Configuration files in `py/configs/` contain all hyperparameters needed to reproduce our results.

**Datasets:**
- CIFAR-10: 512 batch size, 4 H100 GPUs, ~400k steps
- CelebA-64: 512 batch size, 4 H100 GPUs, ~400k steps
- AFHQ-64: 256 batch size, 4 H100 GPUs, ~800k steps
- Checker: 100k batch size, 1 A100 GPU, 250k steps

## 📚 Citation

If you find this work useful, please cite:

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

## 🔗 Links

- **Paper**: [ArXiv 2505.18825](https://arxiv.org/abs/2505.18825)
- **NeurIPS 2025**: Proceedings (forthcoming)
- **Authors**:
  - [Nicholas M. Boffi](https://www.andrew.cmu.edu/user/nboffi/) (Carnegie Mellon University)
  - [Michael S. Albergo](https://malbergo.me/) (Harvard University)
  - [Eric Vanden-Eijnden](https://www.math.nyu.edu/~eve2/) (Courant Institute)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the broader generative modeling community for inspiration and discussions, and the JAX/Flax teams for their excellent framework. Special thanks to the Kempner Institute and Albergo Lab for computational resources.

---

**Questions?** Open an issue or reach out to the authors. We're excited to see what you build with flow maps!
