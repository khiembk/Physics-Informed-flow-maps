"""
PDE flow-map training launcher (single GPU).

Baseline : linear interpolant + diagonal flow-matching loss.
Phase 2  : physics-informed path from frozen phi (set --phi_ckpt).

Usage:
    # Fresh start
    python py/launchers/pde_learn.py \\
        --cfg_path         configs.sw_baseline \\
        --dataset_location /scratch/user/u.kt348068/physics_informedPDE \\
        --output_folder    /scratch/user/u.kt348068/physics_informedPDE/runs

    # Resume from checkpoint
    python py/launchers/pde_learn.py \\
        --cfg_path         configs.sw_baseline \\
        --dataset_location /scratch/user/u.kt348068/physics_informedPDE \\
        --output_folder    /scratch/user/u.kt348068/physics_informedPDE/runs \\
        --continue_from    /scratch/.../runs/sw_baseline_linear/params_step10000.npz
"""

import argparse
import importlib
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.flow_map    import initialize_flow_map
from common.interpolant import setup_interpolant
from common.losses      import diagonal_term


# ---------------------------------------------------------------------------
# Data
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
            yield {"x0": x0s[b], "xT": xTs[b]}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ckpt(path, params, ema_params, step):
    flat,     _ = ravel_pytree(params)
    flat_ema, _ = ravel_pytree(ema_params)
    np.savez_compressed(path,
                        params=np.array(flat),
                        ema_params=np.array(flat_ema),
                        step=np.array(step))
    print(f"  [ckpt] saved {path}  (step {step})")


def load_ckpt(path, ref_params):
    """Returns (params, ema_params, start_step)."""
    d = np.load(path)
    _, unravel = ravel_pytree(ref_params)
    params     = unravel(jnp.array(d["params"]))
    ema_params = unravel(jnp.array(d["ema_params"]
                                    if "ema_params" in d else d["params"]))
    step = int(d["step"]) if "step" in d else 0
    print(f"  [ckpt] loaded {path}  (step {step})")
    return params, ema_params, step


# ---------------------------------------------------------------------------
# Training step: diagonal flow matching, vmapped over batch
# ---------------------------------------------------------------------------

def make_train_step(net, interp):
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def per_sample(params, x0, x1, t, rng_key):
        return diagonal_term(params, x0, x1, None, t, {"dropout": rng_key},
                             interp=interp, X=net)

    @jax.jit
    def train_step(params, ema_params, opt_state, optimizer, x0, x1, t, rng):
        keys = jax.random.split(rng, x0.shape[0])

        def loss_fn(p):
            return jnp.mean(per_sample(p, x0, x1, t, keys))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        ema_new = jax.tree_util.tree_map(
            lambda e, p: 0.999 * e + 0.001 * p, ema_params, params_new
        )
        return params_new, ema_new, opt_state_new, loss, grad_norm

    return train_step


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

    # Data
    system = cfg.problem.system
    x0_tr, xT_tr = load_split(args.dataset_location, system, "train")
    print(f"Train   : {x0_tr.shape}")

    bs = cfg.optimization.bs
    train_iter = make_iterator(x0_tr, xT_tr, bs, seed=cfg.training.seed)

    # Model
    C, H, W = cfg.problem.C, cfg.problem.H, cfg.problem.W
    prng = jax.random.PRNGKey(cfg.training.seed)
    net, params, prng = initialize_flow_map(cfg.network, jnp.zeros((C, H, W)), prng)
    ema_params = params

    # Restore checkpoint
    start_step = 0
    if args.continue_from:
        params, ema_params, start_step = load_ckpt(args.continue_from, params)

    # Interpolant
    interp = setup_interpolant(cfg)

    # Optimizer
    total = cfg.optimization.total_steps
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optimization.learning_rate,
        warmup_steps=cfg.optimization.warmup_steps,
        decay_steps=total,
        end_value=cfg.optimization.learning_rate * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimization.clip),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(params)

    # Compile train step
    train_step = make_train_step(net, interp)

    # Output
    run_dir = os.path.join(args.output_folder, cfg.logging.wandb_name)
    os.makedirs(run_dir, exist_ok=True)

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

    remaining = total - start_step
    print(f"\nBaseline: {remaining} steps remaining  "
          f"(start={start_step}  total={total})")
    print(f"  bs={bs}  lr={cfg.optimization.learning_rate}"
          f"  interp={cfg.problem.interp_type}")

    t0 = time.time()
    for step in range(start_step + 1, total + 1):
        batch = next(train_iter)
        x0 = jnp.array(batch["x0"])
        xT = jnp.array(batch["xT"])
        prng, key_t, key_d = jax.random.split(prng, 3)
        t_samp = jax.random.uniform(key_t, (bs,),
                                     minval=cfg.problem.tmin,
                                     maxval=cfg.problem.tmax)

        params, ema_params, opt_state, loss, grad_norm = train_step(
            params, ema_params, opt_state, optimizer, x0, xT, t_samp, key_d
        )

        if step % cfg.logging.log_freq == 0:
            lr_val = float(lr_schedule(step))
            log = {
                "step":      step,
                "loss":      float(loss),
                "grad_norm": float(grad_norm),
                "lr":        lr_val,
                "elapsed_s": time.time() - t0,
            }
            print(f"  step {step:7d} | loss={log['loss']:.4e}"
                  f"  |g|={log['grad_norm']:.2e}"
                  f"  lr={lr_val:.2e}"
                  f"  {log['elapsed_s']:.0f}s")
            if use_wandb:
                wandb.log(log, step=step)

        if step % cfg.logging.save_freq == 0 or step == total:
            save_ckpt(os.path.join(run_dir, f"params_step{step}.npz"),
                      params, ema_params, step)

    if use_wandb:
        wandb.finish()
    print(f"\nDone. {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
