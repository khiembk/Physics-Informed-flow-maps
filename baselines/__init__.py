"""
baselines/ — Physics-Constrained Flow Map inference applied to pretrained models.

PCFM (Physics-Constrained Flow Matching, arXiv:2506.04171) projects the state
to the constraint manifold at every Euler integration step, enforcing hard
constraints during inference without retraining.

We apply PCFM to our pretrained LINEAR BASELINE flow map v_θ:
    x_{t+dt} = x_t + dt * v_θ(x_t, t, t)        # unconstrained Euler step
    x_{t+dt} = project(x_{t+dt})                 # constraint projection

Per-system projections (all exact, no Newton-Raphson needed for these constraints):
    NS        : Leray projection (spectral)    → ∇·u = 0 exactly
    MHD       : Leray projection x2            → ∇·u = ∇·B = 0 exactly
    SW        : clip η ≥ 0                     → height positivity
    Multiphase: project to probability simplex → S_w+S_o=1, S_i∈[0,1]
    Euler     : clip ρ>0, adjust E for p>0     → pressure positivity

Usage:
    from baselines.pcfm_inference import make_pcfm_euler_fn, PROJECT_FNS
    fn = make_pcfm_euler_fn(net, params, system="navier_stokes_2d", n_steps=10)
    xT_pred = fn(x0_batch)
"""

from .pcfm_inference import make_pcfm_euler_fn, PROJECT_FNS
