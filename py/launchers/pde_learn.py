"""
PDE flow-map training launcher (single GPU).

Works for both:
  - Baseline  : linear interpolant, diagonal flow-matching loss
  - Phase 2   : physics-informed path from frozen phi (future)

Usage:
    python py/launchers/pde_learn.py \\
        --cfg_path          configs.sw_baseline \\
        --dataset_location  /scratch/user/u.kt348068/physics_informedPDE \\
        --output_folder     /scratch/user/u.kt348068/physics_informedPDE/runs

The model learns:
    v_theta(x_t, t, t) ≈  d/dt [ interpolant(t, x0, xT) ]

For baseline  : interpolant = (1-t)*x0 + t*xT
For Phase 2   : interpolant = physics-informed path from phi
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
from common.state_utils import EMATrainState


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(dataset_location, system, split):
    path = os.path.join(dataset_location, system, f"{split}.npz")
    data = np.load(path)
    return data["x0"].astype(np.float32), data["xT"].astype(np.float32)


def make_iterator(x0s, xTs, bs, seed=0):
    """Infinite iterator: yields dict with x0, xT of shape (bs, C, H, W)."""
    N   = x0s.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    while True:
        rng.shuffle(idx)
        for start in range(0, N - bs + 1, bs):
            b = idx[start:start + bs]
            yield {"x0": x0s[b], "xT": xTs[b]}


# ---------------------------------------------------------------------------
# Loss: diagonal flow matching (baseline), vmapped over batch
# ---------------------------------------------------------------------------

def make_train_step(net, interp, optimizer, cfg):
    """Returns a jit-compiled train_step function."""

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def per_sample_loss(params, x0, x1, t, rng_key):
        rng = {"dropout": rng_key}
        return diagonal_term(params, x0, x1, None, t, rng,
                             interp=interp, X=net)

    @jax.jit
    def train_step(state, x0, x1, t, rng):
        dropout_keys = jax.random.split(rng, x0.shape[0])

        def loss_fn(params):
            losses = per_sample_loss(params, x0, x1, t, dropout_keys)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        # gradient norm for logging
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
        )
        state = state.apply_gradients(grads=grads)
        return state, loss, grad_norm

    return train_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",         required=True)
    parser.add_argument("--dataset_location", required=True)
    parser.add_argument("--output_folder",    required=True)
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
    x0_te, xT_te = load_split(args.dataset_location, system, "test")
    print(f"Train: {x0_tr.shape}   Test: {x0_te.shape}")

    bs = cfg.optimization.bs
    train_iter = make_iterator(x0_tr, xT_tr, bs, seed=cfg.training.seed)

    # Model
    C, H, W = cfg.problem.C, cfg.problem.H, cfg.problem.W
    ex_input = jnp.zeros((C, H, W))
    prng = jax.random.PRNGKey(cfg.training.seed)
    net, params, prng = initialize_flow_map(cfg.network, ex_input, prng)

    # Interpolant
    interp = setup_interpolant(cfg)

    # Optimizer
    total = cfg.optimization.total_steps
    warmup = cfg.optimization.warmup_steps
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optimization.learning_rate,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=cfg.optimization.learning_rate * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimization.clip),
        optax.adam(lr_schedule),
    )

    # EMA state: use flax TrainState for convenience
    import flax.training.train_state as train_state

    class TrainStateWithEMA(train_state.TrainState):
        ema_params: dict

    state = TrainStateWithEMA.create(
        apply_fn=net.apply,
        params=params,
        tx=optimizer,
        ema_params=params,
    )

    # Compile train step
    train_step = make_train_step(net, interp, optimizer, cfg)

    # Output
    run_dir = os.path.join(args.output_folder, cfg.logging.wandb_name)
    os.makedirs(run_dir, exist_ok=True)

    # W&B
    try:
        import wandb
        wandb.init(project=cfg.logging.wandb_project,
                   name=cfg.logging.wandb_name,
                   entity=cfg.logging.wandb_entity,
                   config=cfg.to_dict())
        use_wandb = True
    except Exception:
        use_wandb = False
        print("W&B unavailable — stdout only.")

    # EMA decay
    ema_decay = 0.999

    print(f"\nStarting training: {total} steps  bs={bs}  lr={cfg.optimization.learning_rate}")

    tstart = time.time()
    for step in range(1, total + 1):
        batch = next(train_iter)
        x0  = jnp.array(batch["x0"])
        xT  = jnp.array(batch["xT"])
        # sample t ~ U(tmin, tmax) per sample
        prng, key_t, key_d = jax.random.split(prng, 3)
        t   = jax.random.uniform(
            key_t, (bs,),
            minval=cfg.problem.tmin, maxval=cfg.problem.tmax,
        )

        state, loss, grad_norm = train_step(state, x0, xT, t, key_d)

        # EMA update
        ema_new = jax.tree_util.tree_map(
            lambda e, p: ema_decay * e + (1 - ema_decay) * p,
            state.ema_params, state.params,
        )
        state = state.replace(ema_params=ema_new)

        if step % cfg.logging.log_freq == 0:
            lr_val = float(lr_schedule(step))
            log = {
                "step":      step,
                "loss":      float(loss),
                "grad_norm": float(grad_norm),
                "lr":        lr_val,
                "elapsed_s": time.time() - tstart,
            }
            print(f"  step {step:7d} | loss={log['loss']:.4e}"
                  f"  |g|={log['grad_norm']:.2e}"
                  f"  lr={lr_val:.2e}"
                  f"  {log['elapsed_s']:.0f}s")
            if use_wandb:
                wandb.log(log, step=step)

        if step % cfg.logging.save_freq == 0 or step == total:
            ckpt = os.path.join(run_dir, f"params_step{step}.npz")
            flat, _ = ravel_pytree(state.ema_params)
            np.savez_compressed(ckpt, params=np.array(flat))
            print(f"  Saved {ckpt}")

    if use_wandb:
        wandb.finish()
    print(f"\nDone. Total: {time.time()-tstart:.0f}s")


if __name__ == "__main__":
    main()
