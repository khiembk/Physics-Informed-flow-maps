"""
PDE right-hand-side operators for Phase 1 physics residual loss.

Each function computes the expected time derivative of the state under the PDE:
    rhs(x) = dx/dt   (what the PDE says the state should evolve as)

The physics residual is then:
    R(x_t, v_t) = v_t - rhs(x_t)    (path velocity minus PDE prediction)
    L_phy = E[w(t) * ||R||^2]

All functions work on JAX arrays, single sample (no batch dim), shape (C, H, W).
They are vmap-safe and jit-compatible.
"""

import functools
import jax
import jax.numpy as jnp
from ml_collections import config_dict


# ---------------------------------------------------------------------------
# Spectral helpers (JAX, single-sample, shape (H, W) or (C, H, W))
# ---------------------------------------------------------------------------

def _rfft2(x): return jnp.fft.rfft2(x)
def _irfft2(x, H, W): return jnp.fft.irfft2(x, s=(H, W))

def _build_kgrids(H, W):
    """Returns (Ky, Kx) of shape (H, W//2+1) as JAX arrays."""
    kx = jnp.fft.rfftfreq(W) * W
    ky = jnp.fft.fftfreq(H) * H
    Ky, Kx = jnp.meshgrid(ky, kx, indexing='ij')
    return Ky, Kx

def _div(Fx, Fy, Ky, Kx, H, W):
    """Spectral divergence: ∂_x Fx + ∂_y Fy."""
    return _irfft2(1j * Kx * _rfft2(Fx) + 1j * Ky * _rfft2(Fy), H, W)

def _laplacian(f, K2, H, W):
    """Spectral Laplacian Δf."""
    return _irfft2(-K2 * _rfft2(f), H, W)


# ---------------------------------------------------------------------------
# A. Shallow water  —  state: [eta, m_x, m_y]
# ---------------------------------------------------------------------------

def shallow_water_rhs(state: jnp.ndarray, Ky, Kx, K2, H: int, W: int,
                       g: float = 1.0, nu: float = 0.002) -> jnp.ndarray:
    """
    RHS of the shallow water equations.

    state: (3, H, W)  [eta, m_x, m_y]
    returns: d(state)/dt  of same shape
    """
    eta = state[0]
    mx  = state[1]
    my  = state[2]

    eta_s = jnp.maximum(eta, 1e-3)
    ux = mx / eta_s
    uy = my / eta_s

    # Conservative fluxes
    Fx_eta, Fy_eta = mx, my
    Fx_mx  = mx * ux + 0.5 * g * eta ** 2
    Fy_mx  = mx * uy
    Fx_my  = my * ux
    Fy_my  = my * uy + 0.5 * g * eta ** 2

    d_eta = -_div(Fx_eta, Fy_eta, Ky, Kx, H, W)
    d_mx  = -_div(Fx_mx,  Fy_mx,  Ky, Kx, H, W) + nu * _laplacian(mx, K2, H, W)
    d_my  = -_div(Fx_my,  Fy_my,  Ky, Kx, H, W) + nu * _laplacian(my, K2, H, W)

    return jnp.stack([d_eta, d_mx, d_my])


# ---------------------------------------------------------------------------
# B. Navier-Stokes / Boussinesq  —  state: [c, u_x, u_y]
# ---------------------------------------------------------------------------

def ns_boussinesq_rhs(state: jnp.ndarray, Ky, Kx, K2, H: int, W: int,
                       nu: float = 0.001, kappa: float = 0.0005) -> jnp.ndarray:
    """
    RHS using vorticity-streamfunction (vorticity derived from stored velocity).

    state: (3, H, W)  [c, u_x, u_y]
    returns: d(state)/dt
    """
    c  = state[0]
    ux = state[1]
    uy = state[2]

    # Vorticity from velocity: omega = d_x uy - d_y ux
    ux_hat = _rfft2(ux)
    uy_hat = _rfft2(uy)
    omega = _irfft2(1j * Kx * uy_hat - 1j * Ky * ux_hat, H, W)

    # Advection terms
    c_hat = _rfft2(c)
    o_hat = _rfft2(omega)

    adv_c     = ux * _irfft2(1j * Kx * c_hat, H, W) + uy * _irfft2(1j * Ky * c_hat, H, W)
    adv_omega = ux * _irfft2(1j * Kx * o_hat, H, W) + uy * _irfft2(1j * Ky * o_hat, H, W)

    # Vorticity RHS: -adv + nu*Delta + d_x c  (buoyancy)
    d_c     = -adv_c     + kappa * _laplacian(c,     K2, H, W)
    d_omega = -adv_omega + nu    * _laplacian(omega, K2, H, W) \
              + _irfft2(1j * Kx * c_hat, H, W)

    # Convert d_omega/dt back to d_u/dt via streamfunction
    K2s = jnp.where(K2 == 0, 1.0, K2)
    psi_hat = _rfft2(omega) / K2s
    psi_hat = psi_hat.at[0, 0].set(0.0)
    d_psi_hat = _rfft2(d_omega) / K2s
    d_psi_hat = d_psi_hat.at[0, 0].set(0.0)

    d_ux = _irfft2(1j * Ky * d_psi_hat, H, W)
    d_uy = _irfft2(-1j * Kx * d_psi_hat, H, W)

    return jnp.stack([d_c, d_ux, d_uy])


# ---------------------------------------------------------------------------
# C. MHD  —  state: [u_x, u_y, B_x, B_y]
# ---------------------------------------------------------------------------

def mhd_rhs(state: jnp.ndarray, Ky, Kx, K2, H: int, W: int,
             nu: float = 0.001, eta: float = 0.001) -> jnp.ndarray:
    """
    RHS of incompressible MHD in terms of [u_x, u_y, B_x, B_y].
    Uses vorticity/vector-potential formulation internally.

    state: (4, H, W)
    returns: d(state)/dt
    """
    ux = state[0]; uy = state[1]
    Bx = state[2]; By = state[3]

    K2s = jnp.where(K2 == 0, 1.0, K2)

    # Vorticity and current from velocity / magnetic field
    ux_h = _rfft2(ux); uy_h = _rfft2(uy)
    Bx_h = _rfft2(Bx); By_h = _rfft2(By)

    omega = _irfft2(1j*Kx*uy_h - 1j*Ky*ux_h, H, W)   # curl u
    j     = _irfft2(1j*Kx*By_h - 1j*Ky*Bx_h, H, W)   # curl B

    def _jacobian(f, g):
        fh, gh = _rfft2(f), _rfft2(g)
        fx = _irfft2(1j*Kx*fh, H, W); fy = _irfft2(1j*Ky*fh, H, W)
        gx = _irfft2(1j*Kx*gh, H, W); gy = _irfft2(1j*Ky*gh, H, W)
        return fx*gy - fy*gx

    # Stream functions
    psi_h = _rfft2(omega) / K2s; psi_h = psi_h.at[0,0].set(0.0)
    A_h   = _rfft2(j)     / K2s; A_h   = A_h.at[0,0].set(0.0)
    psi   = _irfft2(psi_h, H, W)
    A     = _irfft2(A_h,   H, W)

    d_omega = _jacobian(A, j) - _jacobian(psi, omega) + nu  * _laplacian(omega, K2, H, W)
    d_A     = -_jacobian(psi, A)                       + eta * _laplacian(A,     K2, H, W)

    # d_omega -> d_psi -> d_u
    d_psi_h = _rfft2(d_omega) / K2s; d_psi_h = d_psi_h.at[0,0].set(0.0)
    d_ux = _irfft2(1j*Ky*d_psi_h, H, W)
    d_uy = _irfft2(-1j*Kx*d_psi_h, H, W)

    # d_A -> d_B
    d_A_h = _rfft2(d_A)
    d_Bx  = _irfft2(1j*Ky*d_A_h, H, W)
    d_By  = _irfft2(-1j*Kx*d_A_h, H, W)

    return jnp.stack([d_ux, d_uy, d_Bx, d_By])


# ---------------------------------------------------------------------------
# D. Multiphase  —  state: [P, S_w, S_o]
# ---------------------------------------------------------------------------

def multiphase_rhs(state: jnp.ndarray, Ky, Kx, K2, H: int, W: int,
                    **kwargs) -> jnp.ndarray:
    """
    RHS for simplified multiphase: velocity is encoded in P (stream function),
    only S_w evolves via Buckley-Leverett.

    state: (3, H, W)  [P, S_w, S_o]
    returns: d(state)/dt  (dP/dt=0, dSw/dt, dSo/dt)
    """
    P  = state[0]   # static stream function (velocity encoded here)
    Sw = state[1]
    # Recover velocity from P as stream function
    P_h = _rfft2(P)
    ux  = _irfft2(1j * Ky * P_h, H, W)
    uy  = _irfft2(-1j * Kx * P_h, H, W)

    # Fractional flow fw(S) = S^2 / (S^2 + (1-S)^2)
    Sw2 = Sw * Sw
    fw  = Sw2 / (Sw2 + (1.0 - Sw) ** 2 + 1e-12)

    # Flux divergence: div(fw * u)
    d_Sw = -_div(fw * ux, fw * uy, Ky, Kx, H, W)
    d_So = -d_Sw   # So = 1 - Sw

    return jnp.stack([jnp.zeros_like(P), d_Sw, d_So])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

RESIDUAL_FNS = {
    "shallow_water_2d": shallow_water_rhs,
    "navier_stokes_2d": ns_boussinesq_rhs,
    "mhd_2d":           mhd_rhs,
    "multiphase_2d":    multiphase_rhs,
}


def get_residual_fn(system: str, cfg: dict, H: int, W: int):
    """Return a callable rhs(state) -> d(state)/dt for the given system."""
    import numpy as np
    kx_np = np.fft.rfftfreq(W) * W
    ky_np = np.fft.fftfreq(H) * H
    Ky_np, Kx_np = np.meshgrid(ky_np, kx_np, indexing='ij')
    K2_np = Kx_np**2 + Ky_np**2

    Ky = jnp.array(Ky_np)
    Kx = jnp.array(Kx_np)
    K2 = jnp.array(K2_np)

    fn = RESIDUAL_FNS[system]
    pde_kwargs = {k: v for k, v in cfg.items()
                  if k in ("nu", "kappa", "eta", "g")}

    return functools.partial(fn, Ky=Ky, Kx=Kx, K2=K2, H=H, W=W, **pde_kwargs)
