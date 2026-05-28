"""
Phase 1 training: learn physics-informed path encoding phi.

Usage:
    # Fresh start
    python py/launchers/phase1_learn.py \\
        --cfg_path         configs.sw_phase1 \\
        --dataset_location /scratch/user/u.kt348068/physics_informedPDE \\
        --output_folder    /scratch/user/u.kt348068/physics_informedPDE/checkpoints

    # Resume from checkpoint
    python py/launchers/phase1_learn.py \\
        --cfg_path         configs.sw_phase1 \\
        --dataset_location /scratch/user/u.kt348068/physics_informedPDE \\
        --output_folder    /scratch/user/u.kt348068/physics_informedPDE/checkpoints \\
        --continue_from    /scratch/.../checkpoints/phase1_checkpoints/sw/phi_step5000.npz
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

from common.path_encoding    import initialize_path_encoding
from common.phase1_loss      import make_phase1_loss, train_step
from common.physics_residuals import get_residual_fn


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(dataset_location, system, split):
    path = os.path.join(dataset_location, system, f"{split}.npz")
    d = np.load(path)
    return d["x0"].astype(np.float32), d["xT"].astype(np.float32)


def make_iterator(x0s, xTs, bs, seed=0):
    N = x0s.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    while True:
        rng.shuffle(idx)
        for start in range(0, N - bs + 1, bs):
            b = idx[start:start + bs]
            t = rng.uniform(0.0, 1.0, size=(bs,)).astype(np.float32)
            yield {"x0": x0s[b], "xT": xTs[b], "t": t}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(path, params, step):
    flat, _ = ravel_pytree(params)
    np.savez_compressed(path, params=np.array(flat), step=np.array(step))
    print(f"  [ckpt] saved {path}  (step {step})")


def load_ckpt(path, ref_params):
    """Restore params from NPZ; returns (params, start_step)."""
    d = np.load(path)
    flat = jnp.array(d["params"])
    _, unravel = ravel_pytree(ref_params)
    step = int(d["step"]) if "step" in d else 0
    print(f"  [ckpt] loaded {path}  (step {step})")
    return unravel(flat), step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",         required=True)
    parser.add_argument("--dataset_location", required=True)
    parser.add_argument("--output_folder",    required=True)
    parser.add_argument("--continue_from",    default=None,
                        help="Path to .npz checkpoint to resume from")
    args = parser.parse_args()

    module = importlib.import_module(args.cfg_path)
    cfg = module.get_config(
        dataset_location=args.dataset_location,
        output_folder=args.output_folder,
    )

    print(f"Devices : {jax.devices()}")
    print(f"System  : {cfg.problem.system}")
    print(f"Config  : {args.cfg_path}")

    system = cfg.problem.system
    x0_train, xT_train = load_split(args.dataset_location, system, "train")
    print(f"Train   : {x0_train.shape}")

    train_iter = make_iterator(x0_train, xT_train, cfg.optimization.bs,
                                seed=cfg.optimization.seed)

    # Model init
    ex_x = jnp.zeros((cfg.problem.C, cfg.problem.H, cfg.problem.W))
    prng = jax.random.PRNGKey(cfg.optimization.seed)
    path_model, phi_params, prng = initialize_path_encoding(cfg, ex_x, prng)

    # Optionally restore from checkpoint
    start_step = 0
    if args.continue_from:
        phi_params, start_step = load_ckpt(args.continue_from, phi_params)

    # PDE residual
    pde_cfg = {k: float(v) for k, v in cfg.problem.items()
               if k in ("g", "nu", "kappa")}
    rhs_fn = get_residual_fn(system, pde_cfg, cfg.problem.H, cfg.problem.W)

    # Spectral grids (for L_sm)
    kx = jnp.array(np.fft.rfftfreq(cfg.problem.W) * cfg.problem.W)
    ky = jnp.array(np.fft.fftfreq(cfg.problem.H) * cfg.problem.H)
    Ky, Kx = jnp.meshgrid(ky, kx, indexing='ij')

    loss_fn = make_phase1_loss(
        path_model, rhs_fn, Ky, Kx,
        cfg.problem.H, cfg.problem.W,
        w0=cfg.phase1.w0, w_alpha=cfg.phase1.w_alpha,
        lambda_sm=cfg.phase1.lambda_sm,
    )

    # Optimizer
    total_steps = cfg.optimization.total_steps
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optimization.learning_rate,
        warmup_steps=cfg.optimization.warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.optimization.learning_rate * 0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimization.clip),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(phi_params)

    # Checkpoints dir
    ckpt_dir = os.path.join(args.output_folder, "phase1_checkpoints", system)
    os.makedirs(ckpt_dir, exist_ok=True)

    # W&B
    try:
        import wandb
        wandb.init(project=cfg.logging.wandb_project,
                   name=cfg.logging.wandb_name,
                   entity=cfg.logging.wandb_entity,
                   config=cfg.to_dict(),
                   resume="allow" if args.continue_from else None)
        use_wandb = True
    except Exception:
        use_wandb = False

    remaining = total_steps - start_step
    print(f"\nPhase 1: {remaining} steps remaining  "
          f"(start={start_step}  total={total_steps})")
    print(f"  bs={cfg.optimization.bs}  lr={cfg.optimization.learning_rate}"
          f"  lambda_sm={cfg.phase1.lambda_sm}")

    t0 = time.time()
    for step in range(start_step + 1, total_steps + 1):
        batch = next(train_iter)
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        phi_params, opt_state, loss, metrics = train_step(
            phi_params, opt_state, batch_jax, loss_fn, optimizer
        )

        if step % cfg.logging.log_freq == 0:
            lr_val = float(lr_schedule(step))
            log = {
                "step":    step,
                "loss":    float(loss),
                "L_phy":   float(metrics["L_phy"]),
                "L_sm":    float(metrics["L_sm"]),
                "lr":      lr_val,
                "elapsed": time.time() - t0,
            }
            print(f"  step {step:6d} | loss={log['loss']:.4e}"
                  f"  L_phy={log['L_phy']:.4e}"
                  f"  L_sm={log['L_sm']:.4e}"
                  f"  lr={lr_val:.2e}"
                  f"  {log['elapsed']:.0f}s")
            if use_wandb:
                wandb.log(log, step=step)

        if step % cfg.logging.save_freq == 0:
            save_ckpt(os.path.join(ckpt_dir, f"phi_step{step}.npz"),
                      phi_params, step)

    save_ckpt(os.path.join(ckpt_dir, "phi_final.npz"), phi_params, total_steps)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
