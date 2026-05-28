"""
Phase 1 training: learn physics-informed path encoding phi.

Usage:
    python py/launchers/phase1_learn.py \\
        --cfg_path    configs.sw_phase1 \\
        --dataset_location /scratch/user/u.kt348068/physics_informedPDE \\
        --output_folder    /scratch/user/u.kt348068/physics_informedPDE/checkpoints

Trains PhysicsInformedPath(PhiUNet) to minimize:
    L_path = L_phy + lambda_sm * L_sm
on the shallow_water_2d dataset.
"""

import argparse
import importlib
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.path_encoding import setup_path_encoding, initialize_path_encoding
from common.phase1_loss import make_phase1_loss, train_step
from common.physics_residuals import get_residual_fn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(dataset_location: str, system: str, split: str):
    path = os.path.join(dataset_location, system, f"{split}.npz")
    data = np.load(path)
    return data["x0"].astype(np.float32), data["xT"].astype(np.float32)


def make_iterator(x0s, xTs, bs: int, seed: int = 0):
    """Infinite iterator yielding batches dict{x0, xT, t}."""
    N = x0s.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)

    while True:
        rng.shuffle(idx)
        for start in range(0, N - bs + 1, bs):
            batch_idx = idx[start:start + bs]
            t = rng.uniform(0.0, 1.0, size=(bs,)).astype(np.float32)
            yield {
                "x0": x0s[batch_idx],
                "xT": xTs[batch_idx],
                "t":  t,
            }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",         required=True)
    parser.add_argument("--dataset_location", required=True)
    parser.add_argument("--output_folder",    required=True)
    args = parser.parse_args()

    # Load config
    module = importlib.import_module(args.cfg_path)
    cfg = module.get_config(
        dataset_location=args.dataset_location,
        output_folder=args.output_folder,
    )

    print(f"Devices: {jax.devices()}")
    print(f"System:  {cfg.problem.system}")
    print(f"Config:  {args.cfg_path}")

    # Data
    system = cfg.problem.system
    x0_train, xT_train = load_split(args.dataset_location, system, "train")
    x0_test,  xT_test  = load_split(args.dataset_location, system, "test")
    print(f"Train: {x0_train.shape}   Test: {x0_test.shape}")

    train_iter = make_iterator(x0_train, xT_train, cfg.optimization.bs,
                                seed=cfg.optimization.seed)

    # Model
    ex_x = jnp.zeros((cfg.problem.C, cfg.problem.H, cfg.problem.W))
    prng = jax.random.PRNGKey(cfg.optimization.seed)
    path_model, phi_params, prng = initialize_path_encoding(cfg, ex_x, prng)

    # PDE residual function
    pde_cfg = {k: float(v) for k, v in cfg.problem.items()
               if k in ("g", "nu", "kappa", "eta_mhd")}
    rhs_fn = get_residual_fn(system, pde_cfg, cfg.problem.H, cfg.problem.W)

    # Spectral grids for L_sm
    kx = jnp.array(np.fft.rfftfreq(cfg.problem.W) * cfg.problem.W)
    ky = jnp.array(np.fft.fftfreq(cfg.problem.H) * cfg.problem.H)
    Ky, Kx = jnp.meshgrid(ky, kx, indexing='ij')

    # Loss function
    loss_fn = make_phase1_loss(
        path_model, rhs_fn, Ky, Kx,
        cfg.problem.H, cfg.problem.W,
        w0=cfg.phase1.w0,
        w_alpha=cfg.phase1.w_alpha,
        lambda_sm=cfg.phase1.lambda_sm,
    )

    # Optimizer with warmup + cosine decay
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optimization.learning_rate,
        warmup_steps=cfg.optimization.warmup_steps,
        decay_steps=cfg.optimization.total_steps,
        end_value=cfg.optimization.learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimization.clip),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(phi_params)

    # Output directory
    os.makedirs(args.output_folder, exist_ok=True)
    ckpt_dir = os.path.join(args.output_folder, "phase1_checkpoints", system)
    os.makedirs(ckpt_dir, exist_ok=True)

    # W&B (optional)
    try:
        import wandb
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_name,
            entity=cfg.logging.wandb_entity,
            config=cfg.to_dict(),
        )
        use_wandb = True
    except Exception:
        use_wandb = False
        print("W&B not available, logging to stdout only.")

    print(f"\nStarting Phase 1 training for {cfg.optimization.total_steps} steps")
    print(f"  batch_size={cfg.optimization.bs}  lr={cfg.optimization.learning_rate}"
          f"  lambda_sm={cfg.phase1.lambda_sm}")
    print(f"  L_phy weight: w(t) = {cfg.phase1.w0} + {cfg.phase1.w_alpha}*t")

    t0 = time.time()
    for step in range(1, cfg.optimization.total_steps + 1):
        batch = next(train_iter)
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        phi_params, opt_state, loss, metrics = train_step(
            phi_params, opt_state, batch_jax, loss_fn, optimizer
        )

        if step % cfg.logging.log_freq == 0:
            elapsed = time.time() - t0
            lr_val = float(lr_schedule(step))
            log = {
                "step":    step,
                "loss":    float(loss),
                "L_phy":   float(metrics["L_phy"]),
                "L_sm":    float(metrics["L_sm"]),
                "lr":      lr_val,
                "elapsed": elapsed,
            }
            print(f"  step {step:6d} | loss={log['loss']:.4e}"
                  f"  L_phy={log['L_phy']:.4e}"
                  f"  L_sm={log['L_sm']:.4e}"
                  f"  lr={lr_val:.2e}"
                  f"  {elapsed:.0f}s")
            if use_wandb:
                wandb.log(log, step=step)

        if step % cfg.logging.save_freq == 0:
            ckpt_path = os.path.join(ckpt_dir, f"phi_step{step}.npz")
            flat, _ = ravel_pytree(phi_params)
            np.savez_compressed(ckpt_path, params=np.array(flat))
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, "phi_final.npz")
    flat, _ = ravel_pytree(phi_params)
    np.savez_compressed(final_path, params=np.array(flat))
    print(f"\nFinal checkpoint: {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
