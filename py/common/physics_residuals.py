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
# E. Compressible Euler 2D — state: [rho, m_x, m_y, E]
# ---------------------------------------------------------------------------

def euler_2d_constraint(state: jnp.ndarray,
                         gamma: float = 1.4,
                         **kwargs) -> jnp.ndarray:
    """
    Compressible Euler constraints: rho > 0  AND  p > 0.

    p = (gamma-1)(E - |m|^2 / (2*rho))

    R = stack([relu(-rho), relu(-p)])  ->  shape (2, H, W).

    WHY THIS IS GENUINELY NONLINEAR:
    Both rho_0 > 0 and rho_T > 0 => linear interpolant rho_t > 0 (convex).
    But p_t = (gamma-1)(E_t - |m_t|^2/(2*rho_t)) can be NEGATIVE for
    interpolated states even when p_0 > 0 and p_T > 0, because
    |m_t|^2 / (2*rho_t) is convex in (m,rho) -> Jensen violation.
    """
    rho = state[0]
    mx  = state[1]
    my  = state[2]
    E   = state[3]
    rho_s = jnp.maximum(rho, 1e-6)
    p     = (gamma - 1.0) * (E - (mx**2 + my**2) / (2.0 * rho_s))
    return jnp.stack([jnp.maximum(-rho, 0.0),
                      jnp.maximum(-p,   0.0)])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

CONSTRAINT_FNS = {
    "shallow_water_2d":  shallow_water_constraint,
    "navier_stokes_2d":  ns_boussinesq_constraint,
    "mhd_2d":            mhd_constraint,
    "multiphase_2d":     multiphase_constraint,
    "euler_2d":          euler_2d_constraint,
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

    if system == "euler_2d":
        gamma = float(cfg.get("gamma", 1.4))
        return functools.partial(fn, gamma=gamma)

    return fn   # SW and Multiphase need no spectral grids


# ---------------------------------------------------------------------------
# Keep old name as alias so existing imports don't break immediately
# ---------------------------------------------------------------------------
get_residual_fn = get_constraint_fn


# ===========================================================================
# PDE RIGHT-HAND-SIDE FUNCTIONS  (for dynamics mismatch in Phase1-Dynamic)
#
# rhs(x_t) = dx/dt  as given by the PDE equations evaluated at state x_t.
# Used as: R_dyn = v_t - rhs(x_t)
# This provides a LARGE non-zero gradient from step 1 even when phi≈0,
# because v_t = (x1-x0) and rhs(x_t) differ due to nonlinear PDE dynamics.
# ===========================================================================

def _rfft(x, H, W): return jnp.fft.rfft2(x)
def _irfft(x, H, W): return jnp.fft.irfft2(x, s=(H, W))

def _div2(Fx, Fy, Ky, Kx, H, W):
    return _irfft(1j*Kx*_rfft(Fx,H,W) + 1j*Ky*_rfft(Fy,H,W), H, W)

def _lap(f, K2, H, W):
    return _irfft(-K2 * _rfft(f,H,W), H, W)

def _solve_stream(omega, K2, H, W):
    K2s = jnp.where(K2==0, 1.0, K2)
    psi_hat = _rfft(omega,H,W) / K2s
    psi_hat = psi_hat.at[0,0].set(0.0)
    return psi_hat


# ---------------------------------------------------------------------------
# RHS A: Shallow water  —  state [eta, m_x, m_y]
# ---------------------------------------------------------------------------

def shallow_water_rhs(state: jnp.ndarray, Ky, Kx, K2,
                       H: int, W: int,
                       g: float = 1.0, nu: float = 0.002,
                       **kwargs) -> jnp.ndarray:
    eta, mx, my = state[0], state[1], state[2]
    eta_s = jnp.maximum(eta, 1e-3)
    ux = mx / eta_s; uy = my / eta_s

    d_eta = -_div2(mx, my, Ky, Kx, H, W)
    d_mx  = (-_div2(mx*ux + 0.5*g*eta**2, mx*uy, Ky, Kx, H, W)
              + nu * _lap(mx, K2, H, W))
    d_my  = (-_div2(my*ux, my*uy + 0.5*g*eta**2, Ky, Kx, H, W)
              + nu * _lap(my, K2, H, W))
    return jnp.stack([d_eta, d_mx, d_my])


# ---------------------------------------------------------------------------
# RHS B: NS/Boussinesq  —  state [c, u_x, u_y]
# ---------------------------------------------------------------------------

def ns_boussinesq_rhs(state: jnp.ndarray, Ky, Kx, K2,
                       H: int, W: int,
                       nu: float = 0.001, kappa: float = 0.0005,
                       **kwargs) -> jnp.ndarray:
    c, ux, uy = state[0], state[1], state[2]

    # Vorticity from stored velocity
    ux_h = _rfft(ux, H, W); uy_h = _rfft(uy, H, W)
    omega = _irfft(1j*Kx*uy_h - 1j*Ky*ux_h, H, W)

    # Advection
    oh = _rfft(omega, H, W); ch = _rfft(c, H, W)
    adv_o = ux*_irfft(1j*Kx*oh,H,W) + uy*_irfft(1j*Ky*oh,H,W)
    adv_c = ux*_irfft(1j*Kx*ch,H,W) + uy*_irfft(1j*Ky*ch,H,W)

    d_omega = -adv_o + nu*_lap(omega,K2,H,W) + _irfft(1j*Kx*ch,H,W)
    d_c     = -adv_c + kappa*_lap(c,K2,H,W)

    # d_omega -> d_psi -> d_u
    d_psi_h = _solve_stream(d_omega, K2, H, W)
    # actually need: solve -Δ(d_psi) = d_omega
    # d_psi_h = rfft(d_omega)/K2
    d_psi_h = _rfft(d_omega,H,W) / jnp.where(K2==0,1.0,K2)
    d_psi_h = d_psi_h.at[0,0].set(0.0)
    d_ux = _irfft(1j*Ky*d_psi_h, H, W)
    d_uy = _irfft(-1j*Kx*d_psi_h, H, W)
    return jnp.stack([d_c, d_ux, d_uy])


# ---------------------------------------------------------------------------
# RHS C: MHD  —  state [u_x, u_y, B_x, B_y]
# ---------------------------------------------------------------------------

def mhd_rhs(state: jnp.ndarray, Ky, Kx, K2,
             H: int, W: int,
             nu: float = 0.001, eta: float = 0.001,
             **kwargs) -> jnp.ndarray:
    ux, uy, Bx, By = state[0], state[1], state[2], state[3]
    K2s = jnp.where(K2==0, 1.0, K2)

    # Vorticity ω and current j
    ux_h=_rfft(ux,H,W); uy_h=_rfft(uy,H,W)
    Bx_h=_rfft(Bx,H,W); By_h=_rfft(By,H,W)
    omega = _irfft(1j*Kx*uy_h - 1j*Ky*ux_h, H, W)
    j     = _irfft(1j*Kx*By_h - 1j*Ky*Bx_h, H, W)

    # Stream functions
    psi_h = _rfft(omega,H,W)/K2s; psi_h=psi_h.at[0,0].set(0.0)
    A_h   = _rfft(j,H,W)/K2s;     A_h=A_h.at[0,0].set(0.0)
    psi   = _irfft(psi_h, H, W)
    A     = _irfft(A_h,   H, W)

    def _jac(f, g):
        fh=_rfft(f,H,W); gh=_rfft(g,H,W)
        return (_irfft(1j*Kx*fh,H,W)*_irfft(1j*Ky*gh,H,W)
               -_irfft(1j*Ky*fh,H,W)*_irfft(1j*Kx*gh,H,W))

    d_omega = _jac(A,j) - _jac(psi,omega) + nu  * _lap(omega, K2, H, W)
    d_A     = -_jac(psi,A)                 + eta * _lap(A,     K2, H, W)

    d_psi_h = _rfft(d_omega,H,W)/K2s; d_psi_h=d_psi_h.at[0,0].set(0.0)
    d_ux = _irfft(1j*Ky*d_psi_h, H, W)
    d_uy = _irfft(-1j*Kx*d_psi_h, H, W)
    d_A_h = _rfft(d_A, H, W)
    d_Bx = _irfft(1j*Ky*d_A_h, H, W)
    d_By = _irfft(-1j*Kx*d_A_h, H, W)
    return jnp.stack([d_ux, d_uy, d_Bx, d_By])


# ---------------------------------------------------------------------------
# RHS D: Multiphase  —  state [P, S_w, S_o]
# ---------------------------------------------------------------------------

def multiphase_rhs(state: jnp.ndarray, Ky, Kx, K2,
                    H: int, W: int, **kwargs) -> jnp.ndarray:
    P, Sw = state[0], state[1]
    P_h = _rfft(P, H, W)
    ux  = _irfft(1j*Ky*P_h, H, W)
    uy  = _irfft(-1j*Kx*P_h, H, W)
    Sw2 = Sw*Sw; fw = Sw2/(Sw2+(1-Sw)**2+1e-12)
    d_Sw = -_div2(fw*ux, fw*uy, Ky, Kx, H, W)
    return jnp.stack([jnp.zeros_like(P), d_Sw, -d_Sw])


# ---------------------------------------------------------------------------
# RHS E: Compressible Euler  —  state [rho, m_x, m_y, E]
# ---------------------------------------------------------------------------

def euler_2d_rhs(state: jnp.ndarray, Ky, Kx, K2,
                  H: int, W: int,
                  gamma: float = 1.4, nu: float = 0.008,
                  **kwargs) -> jnp.ndarray:
    rho, mx, my, E = state[0], state[1], state[2], state[3]
    rho_s = jnp.maximum(rho, 1e-4)
    ux = mx/rho_s; uy = my/rho_s
    p  = jnp.maximum((gamma-1)*(E-(mx**2+my**2)/(2*rho_s)), 1e-6)

    d_rho = -_div2(mx, my, Ky, Kx, H, W) + nu*_lap(rho, K2, H, W)
    d_mx  = (-_div2(mx*ux+p, mx*uy, Ky, Kx, H, W)
              + nu*_lap(ux, K2, H, W))
    d_my  = (-_div2(my*ux, my*uy+p, Ky, Kx, H, W)
              + nu*_lap(uy, K2, H, W))
    d_E   = (-_div2((E+p)*ux, (E+p)*uy, Ky, Kx, H, W)
              + nu*_lap(E/rho_s, K2, H, W))
    return jnp.stack([d_rho, d_mx, d_my, d_E])


# ---------------------------------------------------------------------------
# RHS Factory
# ---------------------------------------------------------------------------

RHS_FNS = {
    "shallow_water_2d":  shallow_water_rhs,
    "navier_stokes_2d":  ns_boussinesq_rhs,
    "mhd_2d":            mhd_rhs,
    "multiphase_2d":     multiphase_rhs,
    "euler_2d":          euler_2d_rhs,
}


def get_rhs_fn(system: str, cfg: dict, H: int, W: int):
    """Return rhs_fn(state) -> d(state)/dt for dynamics mismatch loss."""
    fn = RHS_FNS[system]
    kx_np = np.fft.rfftfreq(W) * W
    ky_np = np.fft.fftfreq(H) * H
    Ky_np, Kx_np = np.meshgrid(ky_np, kx_np, indexing='ij')
    K2_np = Kx_np**2 + Ky_np**2
    Ky = jnp.array(Ky_np); Kx = jnp.array(Kx_np); K2 = jnp.array(K2_np)
    pde_kwargs = {k: float(v) for k,v in cfg.items()
                  if k in ("g","nu","kappa","eta","gamma")}
    return functools.partial(fn, Ky=Ky, Kx=Kx, K2=K2, H=H, W=W, **pde_kwargs)
