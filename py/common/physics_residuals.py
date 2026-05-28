"""
Physics constraint violation functions for Phase 1 training.

R(x_t) measures how much the intermediate state x_t violates
the physical admissibility constraints of the PDE system.

    L_phy = E_{t,x0,x1} [ w(t) * ||R(x_t)||^2 ]

This is what the paper means by "violation of physics constraints
at state x_t" — NOT a dynamics mismatch (v_t vs rhs).

Per system:

  NS/Boussinesq  R = div(u_t)          shape (H,W)   — divergence of velocity
  MHD            R = [div(u_t),         shape (2,H,W) — both divergences
                       div(B_t)]
  Multiphase     R = [S_w+S_o-1,        shape (3,H,W) — simplex + bounds
                       relu(-S_w)+relu(S_w-1),
                       relu(-S_o)+relu(S_o-1)]
  Shallow water  R = relu(-eta_t)       shape (H,W)   — height positivity

All functions: single sample (no batch dim), shape (C, H, W).
vmap-safe, jit-compatible.
"""

import functools
import numpy as np

import jax.numpy as jnp
from ml_collections import config_dict


# ---------------------------------------------------------------------------
# Spectral helper
# ---------------------------------------------------------------------------

def _div(vx, vy, Ky, Kx, H, W):
    """Spectral divergence ∂_x vx + ∂_y vy. Returns (H, W)."""
    vx_hat = jnp.fft.rfft2(vx)
    vy_hat = jnp.fft.rfft2(vy)
    return jnp.fft.irfft2(1j * Kx * vx_hat + 1j * Ky * vy_hat, s=(H, W))


# ---------------------------------------------------------------------------
# A. Shallow water — state: [eta, m_x, m_y]
# ---------------------------------------------------------------------------

def shallow_water_constraint(state: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    SW constraint: eta >= 0 (water-height positivity).
    R = relu(-eta_t)  ->  shape (H, W).
    Perfect state: R = 0 everywhere.
    """
    eta = state[0]                          # (H, W)
    return jnp.maximum(-eta, 0.0)


# ---------------------------------------------------------------------------
# B. Navier-Stokes / Boussinesq — state: [c, u_x, u_y]
# ---------------------------------------------------------------------------

def ns_boussinesq_constraint(state: jnp.ndarray,
                               Ky, Kx, H: int, W: int,
                               **kwargs) -> jnp.ndarray:
    """
    NS constraint: div(u) = 0.
    R = ∂_x u_x + ∂_y u_y  ->  shape (H, W).
    Perfect state: R = 0 (machine precision via stream-function init).
    """
    ux = state[1]
    uy = state[2]
    return _div(ux, uy, Ky, Kx, H, W)


# ---------------------------------------------------------------------------
# C. MHD — state: [u_x, u_y, B_x, B_y]
# ---------------------------------------------------------------------------

def mhd_constraint(state: jnp.ndarray,
                    Ky, Kx, H: int, W: int,
                    **kwargs) -> jnp.ndarray:
    """
    MHD constraints: div(u) = 0  AND  div(B) = 0.
    R = stack([div(u_t), div(B_t)])  ->  shape (2, H, W).
    Perfect state: both channels = 0.
    """
    div_u = _div(state[0], state[1], Ky, Kx, H, W)
    div_B = _div(state[2], state[3], Ky, Kx, H, W)
    return jnp.stack([div_u, div_B])


# ---------------------------------------------------------------------------
# D. Multiphase — state: [P, S_w, S_o]
# ---------------------------------------------------------------------------

def multiphase_constraint(state: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """
    Multiphase constraints:
      - Simplex:  S_w + S_o = 1      -> simplex residual
      - Bounds:   0 <= S_i <= 1      -> relu violations
    R = stack([simplex, bound_Sw, bound_So])  ->  shape (3, H, W).
    Perfect state: all channels = 0.
    """
    Sw = state[1]
    So = state[2]
    simplex  = Sw + So - 1.0
    bound_Sw = jnp.maximum(-Sw, 0.0) + jnp.maximum(Sw - 1.0, 0.0)
    bound_So = jnp.maximum(-So, 0.0) + jnp.maximum(So - 1.0, 0.0)
    return jnp.stack([simplex, bound_Sw, bound_So])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

CONSTRAINT_FNS = {
    "shallow_water_2d":  shallow_water_constraint,
    "navier_stokes_2d":  ns_boussinesq_constraint,
    "mhd_2d":            mhd_constraint,
    "multiphase_2d":     multiphase_constraint,
}


def get_constraint_fn(system: str, cfg: dict, H: int, W: int):
    """
    Return a callable  constraint_fn(state) -> R  for the given system.
    For spectral systems (NS, MHD) the wavenumber grids are pre-baked.
    """
    fn = CONSTRAINT_FNS[system]

    if system in ("navier_stokes_2d", "mhd_2d"):
        kx_np = np.fft.rfftfreq(W) * W
        ky_np = np.fft.fftfreq(H) * H
        Ky_np, Kx_np = np.meshgrid(ky_np, kx_np, indexing='ij')
        Ky = jnp.array(Ky_np)
        Kx = jnp.array(Kx_np)
        return functools.partial(fn, Ky=Ky, Kx=Kx, H=H, W=W)

    return fn   # SW and Multiphase need no spectral grids


# ---------------------------------------------------------------------------
# Keep old name as alias so existing imports don't break immediately
# ---------------------------------------------------------------------------
get_residual_fn = get_constraint_fn
