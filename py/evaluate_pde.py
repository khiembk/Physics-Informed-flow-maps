"""
Generic evaluation script for all 4 PDE systems.

Metrics:
  Prediction:  Rel L2 global + per channel
  Constraints: system-specific (div, simplex, positivity, mass)

Usage:
    python py/evaluate_pde.py \\
        --system       navier_stokes_2d \\
        --dataset_location /scratch/.../physics_informedPDE \\
        --phase2_ckpt  /scratch/.../runs/navier_stokes_2d_phase2/params_step6000.npz \\
        --baseline_ckpt /scratch/.../runs/navier_stokes_2d_baseline/params_step10000.npz \\
        --n_steps      1 5 10 \\
        --output_json  /scratch/.../results/ns_eval.json
"""

import argparse, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

sys.path.insert(0, os.path.dirname(__file__))
from common.flow_map import initialize_flow_map
from configs.pde_configs import get_phase2_config, get_baseline_config, SYSTEM_PARAMS


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def make_euler_fn(net, params, n_steps):
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)

    def single_step(x, t_pair):
        t, dt = t_pair
        v = net.apply(params, t, x, None, train=False, method="calc_b")
        return x + dt * v, None

    @jax.jit
    def euler(x0):
        t_pairs = jnp.stack([ts[:-1], jnp.diff(ts)], axis=1)
        xT, _ = jax.lax.scan(single_step, x0, t_pairs)
        return xT

    return jax.jit(jax.vmap(euler))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rel_l2_global(pred, true, eps=1e-8):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps))

def rel_l2_channels(pred, true, names, eps=1e-8):
    return {n: float(np.linalg.norm(pred[:,c]-true[:,c]) /
                     (np.linalg.norm(true[:,c]) + eps))
            for c, n in enumerate(names)}

def spectral_divergence_error(vx, vy, H=64, W=64, eps=1e-12):
    """||div(v)||_2 / (||∇v||_2 + eps) — per sample, returns mean."""
    kx = np.fft.rfftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    Ky, Kx = np.meshgrid(ky, kx, indexing='ij')

    vx_h = np.fft.rfft2(vx)   # (N, H, W//2+1)
    vy_h = np.fft.rfft2(vy)
    div  = np.fft.irfft2(1j*Kx*vx_h + 1j*Ky*vy_h, s=(H, W))

    dvxdx = np.fft.irfft2(1j*Kx*vx_h, s=(H,W));  dvxdy = np.fft.irfft2(1j*Ky*vx_h, s=(H,W))
    dvydx = np.fft.irfft2(1j*Kx*vy_h, s=(H,W));  dvydy = np.fft.irfft2(1j*Ky*vy_h, s=(H,W))
    grad_norm = np.sqrt((dvxdx**2+dvxdy**2+dvydx**2+dvydy**2).mean((-2,-1)))
    div_norm  = np.sqrt((div**2).mean((-2,-1)))
    return float((div_norm / (grad_norm + eps)).mean())

def compute_constraints(system, x0, xT_pred, eps=1e-8):
    """Returns dict of constraint metrics for the system."""
    metrics = {}
    if system == "navier_stokes_2d":
        metrics["DivErr_u"] = spectral_divergence_error(xT_pred[:,1], xT_pred[:,2])

    elif system == "mhd_2d":
        metrics["DivErr_u"] = spectral_divergence_error(xT_pred[:,0], xT_pred[:,1])
        metrics["DivErr_B"] = spectral_divergence_error(xT_pred[:,2], xT_pred[:,3])

    elif system == "multiphase_2d":
        Sw = xT_pred[:,1];  So = xT_pred[:,2]
        metrics["SimplexErr"] = float(np.abs(Sw + So - 1.0).mean())
        metrics["BoundErr"]   = float((np.maximum(-Sw,0)+np.maximum(Sw-1,0)+
                                       np.maximum(-So,0)+np.maximum(So-1,0)).mean())
        metrics["InvalidFrac"] = float(((Sw<0)|(Sw>1)|(So<0)|(So>1)).mean())

    elif system == "shallow_water_2d":
        eta0 = x0[:,0];  etaT = xT_pred[:,0]
        metrics["NegHeight"]  = float(np.maximum(-etaT, 0.0).mean())
        metrics["MassErr"]    = float((np.abs(etaT.mean((-2,-1)) - eta0.mean((-2,-1))) /
                                       (np.abs(eta0.mean((-2,-1))) + eps)).mean())
        metrics["InvalidFrac"] = float((etaT < 0).mean())

    return metrics


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg, ckpt_path, label):
    ex_x = jnp.zeros((cfg.problem.C, 64, 64))
    net, ref_params, _ = initialize_flow_map(cfg.network, ex_x, jax.random.PRNGKey(0))
    d    = np.load(ckpt_path)
    key  = "ema_params" if "ema_params" in d else "params"
    _, unravel = ravel_pytree(ref_params)
    params = unravel(jnp.array(d[key]))
    step   = int(d["step"]) if "step" in d else "?"
    print(f"  Loaded {label}: step={step}  ({ckpt_path.split('/')[-1]})")
    return net, params


# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate_model(net, params, x0, xT_true, system, n_steps_list, label):
    results = {"label": label}
    names   = SYSTEM_PARAMS[system]["channels"]
    for n in n_steps_list:
        fn = make_euler_fn(net, params, n)
        fn(jnp.array(x0[:2]))          # warm-up JIT
        t0 = time.time()
        xT_pred = np.array(fn(jnp.array(x0)))
        elapsed = time.time() - t0
        results[f"steps_{n}"] = {
            "rel_l2_global":   rel_l2_global(xT_pred, xT_true),
            "rel_l2_channels": rel_l2_channels(xT_pred, xT_true, names),
            **compute_constraints(system, x0, xT_pred),
            "inference_s": elapsed,
        }
        print(f"    {label} N={n}: RelL2={results[f'steps_{n}']['rel_l2_global']:.4f}  "
              f"{elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system",           required=True)
    parser.add_argument("--dataset_location", required=True)
    parser.add_argument("--phase2_ckpt",      default=None)
    parser.add_argument("--baseline_ckpt",    default=None)
    parser.add_argument("--n_steps",          type=int, nargs="+", default=[1,5,10])
    parser.add_argument("--output_json",      default=None)
    args = parser.parse_args()

    system = args.system
    names  = SYSTEM_PARAMS[system]["channels"]
    print(f"System: {system}  |  Devices: {jax.devices()}\n")

    d = np.load(os.path.join(args.dataset_location, system, "test.npz"))
    x0, xT = d["x0"].astype(np.float32), d["xT"].astype(np.float32)
    print(f"Test set: {x0.shape}\n")

    all_results = {}
    n = args.n_steps[-1]   # representative step count for printing

    # Trivial
    triv_rl2  = rel_l2_global(x0, xT)
    triv_cons = compute_constraints(system, x0, x0)
    triv_ch   = rel_l2_channels(x0, xT, names)
    all_results["trivial"] = {"label": "trivial (x0)",
                               f"steps_1": {"rel_l2_global": triv_rl2,
                                            "rel_l2_channels": triv_ch, **triv_cons}}
    print(f"  Trivial:  RelL2={triv_rl2:.4f}  {triv_cons}")

    if args.phase2_ckpt and os.path.exists(args.phase2_ckpt):
        print(f"\n  Phase 2 (PI path):")
        cfg2 = get_phase2_config(system, args.dataset_location, "")
        net2, p2 = load_model(cfg2, args.phase2_ckpt, "Phase2")
        all_results["phase2"] = evaluate_model(net2, p2, x0, xT, system, args.n_steps, "Phase2")

    if args.baseline_ckpt and os.path.exists(args.baseline_ckpt):
        print(f"\n  Baseline (linear):")
        cfgb = get_baseline_config(system, args.dataset_location, "")
        netb, pb = load_model(cfgb, args.baseline_ckpt, "Baseline")
        all_results["baseline"] = evaluate_model(netb, pb, x0, xT, system, args.n_steps, "Baseline")

    # Print table
    n_str = ", ".join(str(s) for s in args.n_steps)
    print(f"\n{'='*90}")
    print(f"RESULTS — {system}  (N_steps=[{n_str}], test set {x0.shape[0]} samples)")
    print(f"{'='*90}")
    hdr  = f"{'Model':<22} {'Steps':>5}  {'RelL2↓':>8}"
    for nm in names:
        hdr += f"  {nm[:5]:>7}"
    # Constraint columns
    ckeys = list(triv_cons.keys())
    for ck in ckeys:
        hdr += f"  {ck[:8]:>10}"
    print(hdr);  print("-"*len(hdr))

    def row(label, r, ns):
        k = f"steps_{ns}"
        if k not in r: return
        m  = r[k]
        ch = m["rel_l2_channels"]
        s  = f"  {label:<20} {ns:>5}  {m['rel_l2_global']:>8.4f}"
        for nm in names:
            s += f"  {ch.get(nm, 0):>7.4f}"
        for ck in ckeys:
            s += f"  {m.get(ck, 0):>10.2e}"
        print(s)

    for ns in args.n_steps:
        row("Trivial (x0)",   all_results["trivial"],   1)
        if "phase2"   in all_results: row("Phase2 (PI)", all_results["phase2"],   ns)
        if "baseline" in all_results: row("Baseline",    all_results["baseline"], ns)
        if len(args.n_steps) > 1: print()

    print("="*90)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        def to_py(o):
            if isinstance(o, dict): return {k: to_py(v) for k,v in o.items()}
            return round(float(o), 8) if isinstance(o, (float, np.floating)) else o
        with open(args.output_json, "w") as f:
            json.dump(to_py(all_results), f, indent=2)
        print(f"\nSaved: {args.output_json}")

if __name__ == "__main__":
    main()
