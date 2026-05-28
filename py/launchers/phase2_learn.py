"""
Phase 2: train flow map v_theta on the physics-informed path.

Loads frozen phi from Phase 1 checkpoint, pre-computes physics-informed
path targets (x_t, v_t) each step, then trains v_theta to match them.

Loss (diagonal flow matching on physics-informed path):
    L = ||v_theta(x_t, t, t) - v_t||^2

where (x_t, v_t) = frozen_phi.velocity(t, x0, xT).

Usage:
    # Fresh start
    python py/launchers/phase2_learn.py \\
        --cfg_path         configs.sw_phase2 \\
        --dataset_location /scratch/.../physics_informedPDE \\
        --output_folder    /scratch/.../runs \\
        --phi_ckpt         /scratch/.../checkpoints/phase1_checkpoints/shallow_water_2d/phi_step35000.npz

    # Resume
    python py/launchers/phase2_learn.py ... --continue_from /scratch/.../runs/sw_phase2_pi_path/params_step4000.npz
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

from common.path_encoding    import initialize_path_encoding
from common.flow_map         import initialize_flow_map


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
    d = np.load(path)
    _, unravel = ravel_pytree(ref_params)
    params     = unravel(jnp.array(d["params"]))
    ema_params = unravel(jnp.array(d.get("ema_params", d["params"])))
    step = int(d["step"]) if "step" in d else 0
    print(f"  [ckpt] loaded {path}  (step {step})")
    return params, ema_params, step


def load_phi_ckpt(path, phi_model, cfg):
    """Load frozen phi parameters from Phase 1 checkpoint."""
    d = np.load(path)
    flat = jnp.array(d["params"])
    ex_x = jnp.zeros((cfg.problem.C, cfg.problem.H, cfg.problem.W))
    _, ref_params, _ = initialize_path_encoding(cfg, ex_x, jax.random.PRNGKey(0))
    _, unravel = ravel_pytree(ref_params)
    params = unravel(flat)
    step = int(d["step"]) if "step" in d else "?"
    print(f"  [phi]  loaded {path}  (step {step}, FROZEN)")
    return params


# ---------------------------------------------------------------------------
# Phase 2 training functions
# ---------------------------------------------------------------------------

def make_phase2_fns(phi_model, phi_params_frozen, net, optimizer, bs):
    """
    Returns (compute_targets, train_step) functions.

    phi_params_frozen is captured as a closure — never updated by optimizer.
    """

    @jax.jit
    def compute_targets(x0: jnp.ndarray, x1: jnp.ndarray,
                         t: jnp.ndarray):
        """
        Pre-compute physics-informed path targets from frozen phi.
        Returns (x_t, v_t) each of shape (bs, C, H, W).
        """
        def single(x0_i, x1_i, t_i):
            return phi_model.apply(
                phi_params_frozen, t_i, x0_i, x1_i,
                method=phi_model.velocity,
            )
        return jax.vmap(single)(x0, x1, t)

    @jax.jit
    def train_step(params: dict, ema_params: dict, opt_state,
                    x_t: jnp.ndarray, v_t: jnp.ndarray,
                    t: jnp.ndarray, rng: jnp.ndarray):
        """
        One gradient step: flow-match v_theta to frozen phi path velocity.
        x_t, v_t: pre-computed physics-informed targets (bs, C, H, W).
        """
        keys = jax.random.split(rng, bs)

        def loss_fn(p):
            def per_sample(x_ti, v_ti, t_i, rk):
                # v_theta(x_t, t, t): diagonal flow map velocity
                bt = net.apply(p, t_i, x_ti, None,
                                train=True, method="calc_b",
                                rngs={"dropout": rk})
                return jnp.mean((bt - v_ti) ** 2)

            losses = jax.vmap(per_sample)(x_t, v_t, t, keys)
            return jnp.mean(losses)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads))
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_ema = jax.tree_util.tree_map(
            lambda e, p: 0.999 * e + 0.001 * p, ema_params, new_params
        )
        return new_params, new_ema, new_opt_state, loss, grad_norm

    return compute_targets, train_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path",         required=True)
    parser.add_argument("--dataset_location", required=True)
    parser.add_argument("--output_folder",    required=True)
    parser.add_argument("--phi_ckpt",         required=True,
                        help="Path to Phase 1 phi checkpoint (.npz)")
    parser.add_argument("--continue_from",    default=None,
                        help="Path to Phase 2 v_theta checkpoint to resume")
    args = parser.parse_args()

    module = importlib.import_module(args.cfg_path)
    cfg = module.get_config(
        dataset_location=args.dataset_location,
        output_folder=args.output_folder,
        phi_ckpt=args.phi_ckpt,
    )

    print(f"Devices : {jax.devices()}")
    print(f"System  : {cfg.problem.system}")
    print(f"phi ckpt: {args.phi_ckpt}")

    # Data
    system = cfg.problem.system
    x0_tr, xT_tr = load_split(args.dataset_location, system, "train")
    print(f"Train   : {x0_tr.shape}")

    bs = cfg.optimization.bs
    train_iter = make_iterator(x0_tr, xT_tr, bs, seed=cfg.training.seed)

    # Load frozen phi
    prng = jax.random.PRNGKey(cfg.training.seed)
    ex_x = jnp.zeros((cfg.problem.C, cfg.problem.H, cfg.problem.W))
    phi_model, _, prng = initialize_path_encoding(cfg, ex_x, prng)
    phi_params_frozen = load_phi_ckpt(args.phi_ckpt, phi_model, cfg)

    # Init v_theta (flow map)
    net, vtheta_params, prng = initialize_flow_map(cfg.network, ex_x, prng)
    ema_params = vtheta_params
    start_step = 0

    if args.continue_from:
        vtheta_params, ema_params, start_step = load_ckpt(args.continue_from, vtheta_params)

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
    opt_state = optimizer.init(vtheta_params)

    # Build JIT functions
    compute_targets, train_step = make_phase2_fns(
        phi_model, phi_params_frozen, net, optimizer, bs
    )

    # Output directory
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
    print(f"\nPhase 2: {remaining} steps  (start={start_step}  total={total})")
    print(f"  bs={bs}  lr={cfg.optimization.learning_rate}"
          f"  v_theta params: {ravel_pytree(vtheta_params)[0].size:,}")

    t0 = time.time()
    for step in range(start_step + 1, total + 1):
        batch = next(train_iter)
        x0 = jnp.array(batch["x0"])
        xT = jnp.array(batch["xT"])
        prng, key_t, key_d = jax.random.split(prng, 3)
        t_samp = jax.random.uniform(key_t, (bs,),
                                     minval=cfg.problem.tmin,
                                     maxval=cfg.problem.tmax)

        # Step A: compute physics-informed path targets from frozen phi
        x_t, v_t = compute_targets(x0, xT, t_samp)

        # Step B: gradient step on v_theta
        vtheta_params, ema_params, opt_state, loss, grad_norm = train_step(
            vtheta_params, ema_params, opt_state, x_t, v_t, t_samp, key_d
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
            print(f"  step {step:6d} | loss={log['loss']:.4e}"
                  f"  |g|={log['grad_norm']:.2e}"
                  f"  lr={lr_val:.2e}"
                  f"  {log['elapsed_s']:.0f}s")
            if use_wandb:
                wandb.log(log, step=step)

        if step % cfg.logging.save_freq == 0 or step == total:
            save_ckpt(os.path.join(run_dir, f"params_step{step}.npz"),
                      vtheta_params, ema_params, step)

    if use_wandb:
        wandb.finish()
    print(f"\nDone. {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
