"""
Evaluation script: shallow water state-to-state prediction.

Metrics (all on test set, 256 samples):
  Prediction accuracy:
    - Rel L2 global  = ||xT_pred - xT_true||_F / ||xT_true||_F
    - Rel L2 per channel: eta, m_x, m_y

  Physics constraints:
    - NegHeight      = mean(relu(-eta_T_pred))          (should be 0)
    - MassErr        = |mean(eta_T_pred) - mean(eta_0)| / |mean(eta_0)|
    - InvalidFrac    = fraction of grid cells with eta < 0

Inference options: N-step Euler ODE (N=1,5,10) using diagonal velocity v_theta(x_t,t,t).

Models compared:
  - Trivial (x_pred = x0): lower bound
  - Phase 2 (physics-informed path)
  - Baseline (linear interpolant), if checkpoint available

Usage:
    python py/evaluate_sw.py \\
        --dataset_location /scratch/.../physics_informedPDE \\
        --phase2_ckpt      /scratch/.../runs/sw_phase2_pi_path/params_step6000.npz \\
        --baseline_ckpt    /scratch/.../runs/sw_baseline_linear/params_stepXXXX.npz \\
        --n_steps          10 \\
        --output_json      /scratch/.../results/sw_eval.json
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common.flow_map    import initialize_flow_map
from configs.sw_phase2  import get_config as get_phase2_cfg
from configs.sw_baseline import get_config as get_baseline_cfg


# ---------------------------------------------------------------------------
# Inference: N-step Euler ODE using diagonal velocity
# ---------------------------------------------------------------------------

def make_euler_fn(net, params, n_steps):
    """Returns jit-compiled function: x0 (C,H,W) -> xT_pred (C,H,W)."""
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)

    def single_step(x, t_pair):
        t, dt = t_pair
        v = net.apply(params, t, x, None, train=False, method="calc_b")
        return x + dt * v, None

    @jax.jit
    def euler(x0):
        t_pairs = jnp.stack([ts[:-1], jnp.diff(ts)], axis=1)  # (N, 2)
        xT, _ = jax.lax.scan(single_step, x0, t_pairs)
        return xT

    return jax.jit(jax.vmap(euler))  # batched over N samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rel_l2(pred, true, eps=1e-8):
    """Global relative L2 over all channels and spatial dims."""
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps))


def rel_l2_per_channel(pred, true, channel_names, eps=1e-8):
    """Per-channel relative L2. pred/true: (N, C, H, W)."""
    out = {}
    for c, name in enumerate(channel_names):
        p = pred[:, c]
        t = true[:, c]
        out[name] = float(np.linalg.norm(p - t) / (np.linalg.norm(t) + eps))
    return out


def sw_constraints(x0, xT_pred, eps=1e-8):
    """
    Shallow water physics constraint metrics.
    x0:      (N, 3, H, W)  initial state  [eta, mx, my]
    xT_pred: (N, 3, H, W)  predicted final state
    """
    eta_0    = x0[:, 0]           # (N, H, W)
    eta_pred = xT_pred[:, 0]      # (N, H, W)

    # 1. Height positivity: mean(relu(-eta_T))
    neg_height = float(np.maximum(-eta_pred, 0.0).mean())

    # 2. Fraction of cells with eta < 0
    invalid_frac = float((eta_pred < 0).mean())

    # 3. Mass conservation: |mean(eta_T) - mean(eta_0)| / |mean(eta_0)|
    mean_eta_0    = eta_0.mean(axis=(-2, -1))    # (N,)
    mean_eta_pred = eta_pred.mean(axis=(-2, -1))  # (N,)
    mass_err = float(
        np.abs(mean_eta_pred - mean_eta_0).mean() /
        (np.abs(mean_eta_0).mean() + eps)
    )

    return {
        "NegHeight":   neg_height,
        "InvalidFrac": invalid_frac,
        "MassErr":     mass_err,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg, ckpt_path, label):
    """Load FlowMap from checkpoint; returns (net, params)."""
    ex_x = jnp.zeros((cfg.problem.C, cfg.problem.H, cfg.problem.W))
    net, ref_params, _ = initialize_flow_map(cfg.network, ex_x, jax.random.PRNGKey(0))

    d = np.load(ckpt_path)
    # Prefer EMA params if available
    key = "ema_params" if "ema_params" in d else "params"
    flat = jnp.array(d[key])
    _, unravel = ravel_pytree(ref_params)
    params = unravel(flat)
    step = int(d["step"]) if "step" in d else "?"
    print(f"  Loaded {label}: {ckpt_path}  (step {step}, using '{key}')")
    return net, params


# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate_model(net, params, x0, xT_true, channel_names, n_steps_list,
                   label):
    results = {"label": label}

    for n_steps in n_steps_list:
        print(f"    {label}: running {n_steps}-step Euler ...", end=" ", flush=True)
        euler_fn = make_euler_fn(net, params, n_steps)

        # Warm up JIT
        _ = euler_fn(jnp.array(x0[:2]))

        t0 = time.time()
        xT_pred = np.array(euler_fn(jnp.array(x0)))
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")

        # Metrics
        rl2_global = rel_l2(xT_pred, xT_true)
        rl2_chan   = rel_l2_per_channel(xT_pred, xT_true, channel_names)
        constraints = sw_constraints(x0, xT_pred)

        key = f"steps_{n_steps}"
        results[key] = {
            "rel_l2_global": rl2_global,
            "rel_l2_channels": rl2_chan,
            **constraints,
            "inference_s": elapsed,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_location", required=True)
    parser.add_argument("--phase2_ckpt",    default=None)
    parser.add_argument("--baseline_ckpt",  default=None)
    parser.add_argument("--n_steps",        type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--output_json",    default=None)
    args = parser.parse_args()

    print(f"Devices: {jax.devices()}\n")

    # Load test data
    test_path = os.path.join(args.dataset_location, "shallow_water_2d", "test.npz")
    d = np.load(test_path)
    x0_test = d["x0"].astype(np.float32)    # (256, 3, 64, 64)
    xT_test = d["xT"].astype(np.float32)
    channel_names = ["eta", "m_x", "m_y"]
    print(f"Test set: {x0_test.shape}  ({x0_test.shape[0]} samples)\n")

    all_results = {}

    # ---- Trivial baseline: predict xT = x0 ----
    print("  Trivial (x_pred = x0):")
    rl2_trivial = rel_l2(x0_test, xT_test)
    cons_trivial = sw_constraints(x0_test, x0_test)
    rl2_trivial_ch = rel_l2_per_channel(x0_test, xT_test, channel_names)
    all_results["trivial"] = {
        "label": "trivial (x0)",
        "steps_1": {
            "rel_l2_global":    rl2_trivial,
            "rel_l2_channels":  rl2_trivial_ch,
            **cons_trivial,
        }
    }
    print(f"    RelL2={rl2_trivial:.4f}  NegH={cons_trivial['NegHeight']:.2e}"
          f"  MassErr={cons_trivial['MassErr']:.2e}")

    # ---- Phase 2 model ----
    if args.phase2_ckpt and os.path.exists(args.phase2_ckpt):
        print("\n  Phase 2 (physics-informed path):")
        cfg2 = get_phase2_cfg(dataset_location=args.dataset_location,
                               output_folder="", phi_ckpt="")
        net2, params2 = load_model(cfg2, args.phase2_ckpt, "Phase 2")
        res2 = evaluate_model(net2, params2, x0_test, xT_test,
                               channel_names, args.n_steps, "Phase 2")
        all_results["phase2"] = res2
    else:
        print("\n  Phase 2 checkpoint not provided, skipping.")

    # ---- Baseline model ----
    if args.baseline_ckpt and os.path.exists(args.baseline_ckpt):
        print("\n  Baseline (linear interpolant):")
        cfgb = get_baseline_cfg(dataset_location=args.dataset_location,
                                 output_folder="")
        netb, paramsb = load_model(cfgb, args.baseline_ckpt, "Baseline")
        resb = evaluate_model(netb, paramsb, x0_test, xT_test,
                               channel_names, args.n_steps, "Baseline")
        all_results["baseline"] = resb
    else:
        print("\n  Baseline checkpoint not provided, skipping.")

    # ---- Print comparison table ----
    print("\n" + "=" * 80)
    print("RESULTS — Shallow Water 2D  (test set, 256 samples)")
    print("=" * 80)

    n_steps_report = args.n_steps[0] if len(args.n_steps) == 1 else args.n_steps[-1]

    header = f"{'Model':<25} {'Steps':>5}  {'RelL2↓':>9}  {'eta':>8}  {'m_x':>8}  {'m_y':>8}  {'NegH↓':>9}  {'MassErr↓':>10}  {'InvFrac↓':>10}"
    print(header)
    print("-" * len(header))

    def print_row(label, r, n_steps):
        k = f"steps_{n_steps}"
        if k not in r:
            return
        m = r[k]
        ch = m["rel_l2_channels"]
        print(f"  {label:<23} {n_steps:>5}  "
              f"{m['rel_l2_global']:>9.4f}  "
              f"{ch['eta']:>8.4f}  "
              f"{ch['m_x']:>8.4f}  "
              f"{ch['m_y']:>8.4f}  "
              f"{m['NegHeight']:>9.2e}  "
              f"{m['MassErr']:>10.2e}  "
              f"{m['InvalidFrac']:>10.4f}")

    for n in args.n_steps:
        print_row("Trivial (x0)",          all_results["trivial"],  1)
        if "phase2"   in all_results:
            print_row("Phase2 (PI path)",  all_results["phase2"],   n)
        if "baseline" in all_results:
            print_row("Baseline (linear)", all_results["baseline"], n)
        if len(args.n_steps) > 1:
            print()

    print("=" * 80)

    # ---- Save JSON ----
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        # Convert numpy floats for JSON serialisation
        def to_py(obj):
            if isinstance(obj, dict):
                return {k: to_py(v) for k, v in obj.items()}
            if isinstance(obj, (np.floating, float)):
                return round(float(obj), 8)
            return obj

        with open(args.output_json, "w") as f:
            json.dump(to_py(all_results), f, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
