"""
Phase 1 DYNAMIC training: combined constraint violation + dynamics mismatch.

Loss:
    L_phy = w(t) * ( ||R_con(x_t)||²  +  alpha_dyn * ||v_t - rhs(x_t)||² )

where:
    R_con = constraint_fn(x_t)    — algebraic physics constraint
    R_dyn = v_t - rhs(x_t)       — path velocity vs PDE right-hand-side

Key advantage over constraint-only:
    R_dyn is large from step 1 even when phi≈0 (linear path ≠ PDE trajectory),
    solving the dead-gradient problem of the constraint-only approach.

Usage:
    python py/launchers/phase1_dynamic_learn.py \\
        --cfg_path         configs.ns_phase1 \\
        --dataset_location /scratch/.../physics_informedPDE \\
        --output_folder    /scratch/.../checkpoints

    # Resume
    python py/launchers/phase1_dynamic_learn.py ... \\
        --continue_from    /scratch/.../phi_step1500.npz
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
from common.phase1_loss      import make_phase1_dynamic_loss, train_step
from common.physics_residuals import get_constraint_fn, get_rhs_fn


# ---------------------------------------------------------------------------
# Data + Checkpoint helpers  (same as phase1_learn.py)
# ---------------------------------------------------------------------------

def load_split(dataset_location, system, split):
    d = np.load(os.path.join(dataset_location, system, f"{split}.npz"))
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


def save_ckpt(path, params, step):
    flat, _ = ravel_pytree(params)
    np.savez_compressed(path, params=np.array(flat), step=np.array(step))
    print(f"  [ckpt] saved {path}  (step {step})")


def load_ckpt(path, ref_params):
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
    parser.add_argument("--continue_from",    default=None)
    args = parser.parse_args()

    module = importlib.import_module(args.cfg_path)
    cfg = module.get_config(
        dataset_location=args.dataset_location,
        output_folder=args.output_folder,
    )

    print(f"Devices : {jax.devices()}")
    print(f"System  : {cfg.problem.system}")
    print(f"Mode    : Phase1-Dynamic (constraint + dynamics mismatch)")

    system = cfg.problem.system
    x0_tr, xT_tr = load_split(args.dataset_location, system, "train")
    print(f"Train   : {x0_tr.shape}")

    train_iter = make_iterator(x0_tr, xT_tr, cfg.optimization.bs,
                                seed=cfg.optimization.seed)

    # Model init
    ex_x = jnp.zeros((cfg.problem.C, cfg.problem.H, cfg.problem.W))
    prng = jax.random.PRNGKey(cfg.optimization.seed)
    path_model, phi_params, prng = initialize_path_encoding(cfg, ex_x, prng)

    start_step = 0
    if args.continue_from:
        phi_params, start_step = load_ckpt(args.continue_from, phi_params)

    # Constraint + RHS functions
    pde_cfg = {k: float(v) for k, v in cfg.problem.items()
               if k in ("g", "nu", "kappa", "gamma")}
    constraint_fn = get_constraint_fn(system, pde_cfg, cfg.problem.H, cfg.problem.W)
    rhs_fn        = get_rhs_fn(       system, pde_cfg, cfg.problem.H, cfg.problem.W)

    # Spectral grids for L_sm
    kx = jnp.array(np.fft.rfftfreq(cfg.problem.W) * cfg.problem.W)
    ky = jnp.array(np.fft.fftfreq(cfg.problem.H) * cfg.problem.H)
    Ky, Kx = jnp.meshgrid(ky, kx, indexing='ij')

    # Combined loss
    alpha_dyn = float(cfg.phase1.get("alpha_dyn", 1.0))
    loss_fn = make_phase1_dynamic_loss(
        path_model, constraint_fn, rhs_fn, Ky, Kx,
        cfg.problem.H, cfg.problem.W,
        w0=cfg.phase1.w0, w_alpha=cfg.phase1.w_alpha,
        lambda_sm=cfg.phase1.lambda_sm,
        alpha_dyn=alpha_dyn,
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

    # Output
    ckpt_dir = os.path.join(args.output_folder,
                             "phase1_dynamic_checkpoints", system)
    os.makedirs(ckpt_dir, exist_ok=True)

    # W&B
    try:
        import wandb
        wandb.init(project=cfg.logging.wandb_project,
                   name=cfg.logging.wandb_name + "_dynamic",
                   entity=cfg.logging.wandb_entity,
                   config=cfg.to_dict(),
                   resume="allow" if args.continue_from else None)
        use_wandb = True
    except Exception:
        use_wandb = False

    remaining = total_steps - start_step
    print(f"\nPhase1-Dynamic: {remaining} steps  (alpha_dyn={alpha_dyn}  "
          f"lambda_sm={cfg.phase1.lambda_sm})")

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
                "L_con":   float(metrics.get("L_con", 0)),
                "L_dyn":   float(metrics.get("L_dyn", 0)),
                "L_sm":    float(metrics.get("L_sm",  0)),
                "lr":      lr_val,
                "elapsed": time.time() - t0,
            }
            print(f"  step {step:6d} | loss={log['loss']:.4e}"
                  f"  L_con={log['L_con']:.4e}"
                  f"  L_dyn={log['L_dyn']:.4e}"
                  f"  L_sm={log['L_sm']:.4e}"
                  f"  lr={lr_val:.2e}  {log['elapsed']:.0f}s")
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
