"""
Run PCFM inference on all 5 systems and compare with:
  - Trivial (x_pred = x0)
  - Linear baseline (pretrained v_theta, no projection)
  - PCFM+Linear (pretrained v_theta + per-step constraint projection)
  - Phase2 (physics-informed path v_theta, no projection)

Usage:
    python baselines/run_pcfm.py \\
        --dataset_location /scratch/.../physics_informedPDE \\
        --output_json      /scratch/.../results/pcfm_eval.json \\
        --n_steps          10 \\
        --systems          navier_stokes_2d mhd_2d euler_2d
"""

import argparse, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

from common.flow_map       import initialize_flow_map
from configs.pde_configs   import get_baseline_config, get_phase2_config, SYSTEM_PARAMS
from baselines.pcfm_inference import make_pcfm_euler_fn


# ---------------------------------------------------------------------------
# Metrics (same as evaluate_pde.py)
# ---------------------------------------------------------------------------

def rel_l2(pred, true, eps=1e-8):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps))

def rel_l2_channels(pred, true, names, eps=1e-8):
    return {n: float(np.linalg.norm(pred[:,i]-true[:,i]) /
                     (np.linalg.norm(true[:,i]) + eps))
            for i, n in enumerate(names)}

def spectral_div_err(vx, vy, H=64, W=64, eps=1e-12):
    kx = np.fft.rfftfreq(W) * W
    ky = np.fft.fftfreq(H) * H
    Ky, Kx = np.meshgrid(ky, kx, indexing='ij')
    div = np.fft.irfft2(1j*Kx*np.fft.rfft2(vx) + 1j*Ky*np.fft.rfft2(vy), s=(H,W))
    dvx = np.fft.irfft2(1j*Kx*np.fft.rfft2(vx), s=(H,W))
    dvy = np.fft.irfft2(1j*Ky*np.fft.rfft2(vy), s=(H,W))
    dvxd = np.fft.irfft2(1j*Ky*np.fft.rfft2(vx), s=(H,W))
    dvyd = np.fft.irfft2(1j*Kx*np.fft.rfft2(vy), s=(H,W))
    gn = np.sqrt((dvx**2+dvy**2+dvxd**2+dvyd**2).mean((-2,-1)))
    dn = np.sqrt((div**2).mean((-2,-1)))
    return float((dn/(gn+eps)).mean())

def compute_constraints(system, x0, xT_pred, eps=1e-8):
    m = {}
    if system == "navier_stokes_2d":
        m["DivErr_u"] = spectral_div_err(xT_pred[:,1], xT_pred[:,2])
    elif system == "mhd_2d":
        m["DivErr_u"] = spectral_div_err(xT_pred[:,0], xT_pred[:,1])
        m["DivErr_B"] = spectral_div_err(xT_pred[:,2], xT_pred[:,3])
    elif system == "shallow_water_2d":
        eta0, etaT = x0[:,0], xT_pred[:,0]
        m["NegHeight"]  = float(np.maximum(-etaT, 0).mean())
        m["MassErr"]    = float((np.abs(etaT.mean((-2,-1))-eta0.mean((-2,-1))) /
                                  (np.abs(eta0.mean((-2,-1)))+eps)).mean())
        m["InvalidFrac"] = float((etaT < 0).mean())
    elif system == "multiphase_2d":
        Sw, So = xT_pred[:,1], xT_pred[:,2]
        m["SimplexErr"] = float(np.abs(Sw+So-1).mean())
        m["BoundErr"]   = float((np.maximum(-Sw,0)+np.maximum(Sw-1,0)+
                                  np.maximum(-So,0)+np.maximum(So-1,0)).mean())
        m["InvalidFrac"] = float(((Sw<0)|(Sw>1)|(So<0)|(So>1)).mean())
    elif system == "euler_2d":
        rho, mx, my, E = xT_pred[:,0], xT_pred[:,1], xT_pred[:,2], xT_pred[:,3]
        p = (1.4-1)*(E - (mx**2+my**2)/(2*np.maximum(rho,1e-6)))
        m["NegRho"]      = float(np.maximum(-rho, 0).mean())
        m["NegP"]        = float(np.maximum(-p,   0).mean())
        m["InvalidFrac"] = float(((rho<0)|(p<0)).mean())
        # linear interpolant pressure check
        rho0, mx0, my0, E0 = x0[:,0], x0[:,1], x0[:,2], x0[:,3]
        rm = 0.5*rho0+0.5*rho; mm = 0.5*mx0+0.5*mx; mym=0.5*my0+0.5*my; Em=0.5*E0+0.5*E
        p_lin = (1.4-1)*(Em-(mm**2+mym**2)/(2*np.maximum(rm,1e-6)))
        m["NegP_linear"] = float(np.maximum(-p_lin,0).mean())
    return m


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg, ckpt_path):
    ex_x = jnp.zeros((cfg.problem.C, 64, 64))
    net, ref_params, _ = initialize_flow_map(cfg.network, ex_x, jax.random.PRNGKey(0))
    d = np.load(ckpt_path)
    key = "ema_params" if "ema_params" in d else "params"
    _, unravel = ravel_pytree(ref_params)
    params = unravel(jnp.array(d[key]))
    step = int(d.get("step", 0))
    print(f"  Loaded: {os.path.basename(ckpt_path)}  step={step}  ({net.__class__.__name__})")
    return net, params


def make_vanilla_euler(net, params, n_steps):
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)
    def step(x, tp):
        t, dt = tp
        return x + dt * net.apply(params, t, x, None, train=False, method="calc_b"), None
    @jax.jit
    def run(x0):
        xT, _ = jax.lax.scan(step, x0, jnp.stack([ts[:-1], jnp.diff(ts)], 1))
        return xT
    return jax.jit(jax.vmap(run))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_location", required=True)
    parser.add_argument("--output_json",      default=None)
    parser.add_argument("--n_steps",          type=int, default=10)
    parser.add_argument("--systems",          nargs="+",
                        default=["shallow_water_2d","navier_stokes_2d",
                                 "mhd_2d","multiphase_2d","euler_2d"])
    args = parser.parse_args()

    RUNS    = os.path.join(args.dataset_location, "..", "physics_informedPDE", "runs")
    RUNS    = os.path.normpath(os.path.join(args.dataset_location, "runs"))
    # checkpoint name mapping
    BL_NAME = {
        "shallow_water_2d": "sw_baseline_linear",
        "navier_stokes_2d": "navier_stokes_2d_baseline",
        "mhd_2d":           "mhd_2d_baseline",
        "multiphase_2d":    "multiphase_2d_baseline",
        "euler_2d":         "euler_2d_baseline",
    }
    P2_NAME = {
        "shallow_water_2d": "sw_phase2_pi_path",
        "navier_stokes_2d": "navier_stokes_2d_phase2",
        "mhd_2d":           "mhd_2d_phase2",
        "multiphase_2d":    "multiphase_2d_phase2",
        "euler_2d":         "euler_2d_phase2",
    }

    all_results = {}
    print(f"Devices: {jax.devices()}  |  n_steps={args.n_steps}\n")

    for system in args.systems:
        names = SYSTEM_PARAMS[system]["channels"]
        print(f"{'='*70}\n  {system}\n{'='*70}")

        # Load data
        d = np.load(os.path.join(args.dataset_location, system, "test.npz"))
        x0   = d["x0"].astype(np.float32)
        xT   = d["xT"].astype(np.float32)
        x0_j = jnp.array(x0);  xT_j = jnp.array(xT)
        print(f"  Test: {x0.shape}")

        res = {}

        # Trivial
        triv_rl2 = rel_l2(x0, xT)
        triv_ch  = rel_l2_channels(x0, xT, names)
        res["trivial"] = {"label": "Trivial (x0)",
                           f"steps_{args.n_steps}": {
                               "rel_l2_global": triv_rl2,
                               "rel_l2_channels": triv_ch,
                               **compute_constraints(system, x0, x0)}}
        print(f"  Trivial:       RelL2={triv_rl2:.4f}")

        # --- Linear Baseline ---
        bl_ckpt = os.path.join(RUNS, BL_NAME[system], "params_step10000.npz")
        if os.path.exists(bl_ckpt):
            cfgb = get_baseline_config(system, args.dataset_location, "")
            netb, pb = load_model(cfgb, bl_ckpt)
            fn_bl  = make_vanilla_euler(netb, pb, args.n_steps)
            fn_bl(x0_j[:2])   # warmup
            t0 = time.time()
            pred_bl = np.array(fn_bl(x0_j))
            print(f"  Linear:        RelL2={rel_l2(pred_bl,xT):.4f}  {time.time()-t0:.1f}s")
            res["baseline"] = {"label": "Linear baseline",
                                f"steps_{args.n_steps}": {
                                    "rel_l2_global": rel_l2(pred_bl, xT),
                                    "rel_l2_channels": rel_l2_channels(pred_bl, xT, names),
                                    **compute_constraints(system, x0, pred_bl),
                                    "inference_s": time.time()-t0}}

            # --- PCFM + Linear ---
            fn_pcfm = make_pcfm_euler_fn(netb, pb, system,
                                          n_steps=args.n_steps, H=64, W=64)
            fn_pcfm(x0_j[:2])  # warmup
            t0 = time.time()
            pred_pcfm = np.array(fn_pcfm(x0_j))
            print(f"  PCFM+Linear:   RelL2={rel_l2(pred_pcfm,xT):.4f}  {time.time()-t0:.1f}s")
            res["pcfm_linear"] = {"label": "PCFM+Linear",
                                   f"steps_{args.n_steps}": {
                                       "rel_l2_global": rel_l2(pred_pcfm, xT),
                                       "rel_l2_channels": rel_l2_channels(pred_pcfm, xT, names),
                                       **compute_constraints(system, x0, pred_pcfm),
                                       "inference_s": time.time()-t0}}
        else:
            print(f"  [skip baseline — {bl_ckpt} not found]")

        # --- Phase 2 (for reference) ---
        p2_ckpt = os.path.join(RUNS, P2_NAME[system], "params_step6000.npz")
        if os.path.exists(p2_ckpt):
            cfg2 = get_phase2_config(system, args.dataset_location, "", "")
            net2, p2 = load_model(cfg2, p2_ckpt)
            fn_p2  = make_vanilla_euler(net2, p2, args.n_steps)
            fn_p2(x0_j[:2])
            t0 = time.time()
            pred_p2 = np.array(fn_p2(x0_j))
            print(f"  Phase2:        RelL2={rel_l2(pred_p2,xT):.4f}  {time.time()-t0:.1f}s")
            res["phase2"] = {"label": "Phase2",
                              f"steps_{args.n_steps}": {
                                  "rel_l2_global": rel_l2(pred_p2, xT),
                                  "rel_l2_channels": rel_l2_channels(pred_p2, xT, names),
                                  **compute_constraints(system, x0, pred_p2),
                                  "inference_s": time.time()-t0}}

        all_results[system] = res

        # Print comparison
        if "baseline" in res and "pcfm_linear" in res:
            k = f"steps_{args.n_steps}"
            bv  = res["baseline"][k];   pv  = res["pcfm_linear"][k]
            for ck in list(bv.keys()):
                if ck in ("rel_l2_global","rel_l2_channels","inference_s"): continue
                bvc, pvc = bv.get(ck,0), pv.get(ck,0)
                if bvc == 0 and pvc == 0: continue
                d_pct = (pvc-bvc)/(bvc+1e-12)*100
                print(f"    {ck:<12}: Linear={bvc:.3e}  PCFM={pvc:.3e}  Δ={d_pct:+.1f}%")
        print()

    # Save JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        def to_py(o):
            if isinstance(o, dict): return {k: to_py(v) for k,v in o.items()}
            return round(float(o),8) if isinstance(o,(float,np.floating)) else o
        with open(args.output_json, "w") as f:
            json.dump(to_py(all_results), f, indent=2)
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
