"""
Two-phase saturation transport on [0, 2pi]^2.

Saturation equation (Buckley-Leverett):
    d_t Sw + div( fw(Sw) * u ) = 0
    So = 1 - Sw

Velocity u = curl(psi) is fixed (div-free, from a random stream function).
Pressure P = psi  (stream function acts as a proxy for pressure).

Fractional flow:  fw(S) = S^2 / (S^2 + (1-S)^2)   [Corey, M=1]

Numerics: first-order upwind finite volume on uniform grid (conservative).
CFL condition: dt * |u|_max / dx < 1.

State stored: [P, S_w, S_o]   shape (N, 3, H, W)
Constraints:  0 <= S_w <= 1,  S_w + S_o = 1
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.spectral import (
    wavenumbers_2d, k_squared, dealias_mask,
    grf, stream_to_velocity, simplex_error,
)


# ---------------------------------------------------------------------------
# Fractional flow function
# ---------------------------------------------------------------------------

def _fw(S):
    """Corey fractional flow fw(S) = S^2 / (S^2 + (1-S)^2)."""
    S2 = S * S
    om = (1.0 - S) ** 2
    return S2 / (S2 + om + 1e-12)


# ---------------------------------------------------------------------------
# Upwind finite-volume step
# ---------------------------------------------------------------------------

def _upwind_step(Sw, ux, uy, dt, dx, dy):
    """One explicit upwind FV step for d_t Sw + div(fw(Sw)*u) = 0.

    Sw, ux, uy: (N, H, W)
    Returns updated Sw (N, H, W), still in [0,1] if CFL satisfied.
    """
    fw = _fw(Sw)

    # x-direction fluxes at cell interfaces (periodic)
    # Interface (i+1/2): if ux > 0 use left cell, else right cell
    ux_r = np.roll(ux, -1, axis=-1)   # ux at right face = ux[i+1, j] (shifted)

    # Face velocity at i+1/2: use upwind average
    # Standard upwind: F_{i+1/2} = fw(S_i)*max(ux,0) + fw(S_{i+1})*min(ux,0)
    fw_r = np.roll(fw, -1, axis=-1)   # fw at right neighbour
    Fx = fw * np.maximum(ux, 0.0) + fw_r * np.minimum(ux, 0.0)

    # y-direction fluxes
    uy_r = np.roll(uy, -1, axis=-2)
    fw_u = np.roll(fw, -1, axis=-2)
    Fy = fw * np.maximum(uy, 0.0) + fw_u * np.minimum(uy, 0.0)

    # Conservative update: Sw -= dt/dx * (F_{i+1/2} - F_{i-1/2})
    dFx = Fx - np.roll(Fx, 1, axis=-1)
    dFy = Fy - np.roll(Fy, 1, axis=-2)

    Sw_new = Sw - (dt / dx) * dFx - (dt / dy) * dFy

    # Clip to maintain bounds (diffusion from numerical dissipation)
    return np.clip(Sw_new, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(cfg: dict, seed: int) -> dict:
    N    = cfg["N"]
    H    = cfg.get("H", 64)
    W    = cfg.get("W", 64)
    T    = cfg.get("T", 0.1)
    dt   = cfg.get("dt", 0.001)
    grf_alpha_psi = cfg.get("grf_alpha_psi", 4.0)
    grf_alpha_S   = cfg.get("grf_alpha_S",   3.0)
    psi_scale     = cfg.get("psi_scale", 0.5)   # controls velocity magnitude

    rng = np.random.default_rng(seed)
    Ky, Kx = wavenumbers_2d(H, W)
    K2   = k_squared(H, W)
    mask = dealias_mask(H, W)

    dx = 2.0 * np.pi / W
    dy = 2.0 * np.pi / H

    # Fixed velocity field from random stream function (divergence-free)
    psi = psi_scale * grf(N, H, W, alpha=grf_alpha_psi, rng=rng)  # (N, H, W)
    ux, uy = stream_to_velocity(psi, Ky, Kx, H, W)

    # Verify CFL: dt * |u|_max / dx < 1
    u_max = max(np.abs(ux).max(), np.abs(uy).max())
    cfl = dt * u_max / dx
    if cfl > 0.9:
        print(f"  [Multiphase] WARNING: CFL={cfl:.2f} > 0.9; consider reducing dt or psi_scale")

    # Initial saturation: smooth, strictly in (0, 1)
    raw = grf(N, H, W, alpha=grf_alpha_S, rng=rng)   # (N, H, W), ~N(0,1)
    Sw0 = 0.5 + 0.4 * np.tanh(raw)                    # smooth map to (0.1, 0.9)
    So0 = 1.0 - Sw0

    x0 = np.stack([psi, Sw0, So0], axis=1)   # (N, 3, H, W)

    # Time integration (upwind FV)
    Sw = Sw0.copy()
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        Sw = _upwind_step(Sw, ux, uy, dt, dx, dy)

    SoT = 1.0 - Sw
    xT = np.stack([psi, Sw, SoT], axis=1)   # (N, 3, H, W)  (P=psi is static)

    # Constraint checks
    simp = simplex_error(Sw, SoT)
    bound_vio = (np.maximum(-Sw, 0.0) + np.maximum(Sw - 1.0, 0.0)).mean(axis=(-2,-1))
    print(f"  [Multiphase] SimplexErr: {simp.mean():.2e}  BoundViolation: {bound_vio.mean():.2e}")

    return {
        "x0": x0.astype(np.float32),
        "xT": xT.astype(np.float32),
        "t0": 0.0, "tT": float(T), "dt": float(dt),
        "channel_names": ["P", "S_w", "S_o"],
        "params": {"H": H, "W": W, "psi_scale": psi_scale},
    }
