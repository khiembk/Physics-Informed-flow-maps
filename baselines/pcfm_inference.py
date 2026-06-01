"""
PCFM inference: pretrained flow map + per-step constraint projection.

Reference: Physics-Constrained Flow Matching (arXiv:2506.04171)

Algorithm at each Euler step:
    v   = v_theta(x_t, t, t)          # pretrained model (frozen)
    x'  = x_t + dt * v                # unconstrained step
    x'' = project(x')                 # hard constraint projection
    (v_proj = (x'' - x_t) / dt)       # implied projected velocity

Projection functions are exact (analytical) for all our systems —
no Newton-Raphson needed:

    NS:         Leray projection  (spectral, exact, O(N log N))
    MHD:        Leray x2          (spectral, exact)
    SW:         relu clip on η    (pointwise, exact)
    Multiphase: simplex project   (pointwise, exact: clip then renorm)
    Euler:      ρ-clip + E-adjust (pointwise, monotone map)
"""

import numpy as np
import functools
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Spectral helpers (JAX, shape (H, W//2+1) grids)
# ---------------------------------------------------------------------------

def _make_kgrids(H: int, W: int):
    kx = jnp.array(np.fft.rfftfreq(W) * W)   # (W//2+1,)
    ky = jnp.array(np.fft.fftfreq(H) * H)    # (H,)
    Ky, Kx = jnp.meshgrid(ky, kx, indexing='ij')  # (H, W//2+1)
    return Ky, Kx


def _leray(vx: jnp.ndarray, vy: jnp.ndarray,
            Ky: jnp.ndarray, Kx: jnp.ndarray,
            H: int, W: int):
    """Leray projection: (vx,vy) -> divergence-free (px,py).
    Exact via Fourier: p_hat = v_hat - k(k·v_hat)/|k|^2
    """
    vx_hat = jnp.fft.rfft2(vx)
    vy_hat = jnp.fft.rfft2(vy)
    K2     = Kx**2 + Ky**2
    K2s    = jnp.where(K2 == 0, 1.0, K2)
    kdotv  = (Kx * vx_hat + Ky * vy_hat) / K2s
    px_hat = vx_hat - Kx * kdotv
    py_hat = vy_hat - Ky * kdotv
    # zero mean
    px_hat = px_hat.at[0, 0].set(0.0)
    py_hat = py_hat.at[0, 0].set(0.0)
    return (jnp.fft.irfft2(px_hat, s=(H, W)),
            jnp.fft.irfft2(py_hat, s=(H, W)))


# ---------------------------------------------------------------------------
# Per-system projection functions  (single sample, shape (C, H, W))
# ---------------------------------------------------------------------------

def _proj_ns(x: jnp.ndarray, Ky, Kx, H, W) -> jnp.ndarray:
    """NS [c, u_x, u_y]: Leray project velocity channels."""
    c, ux, uy = x[0], x[1], x[2]
    px, py = _leray(ux, uy, Ky, Kx, H, W)
    return jnp.stack([c, px, py])


def _proj_mhd(x: jnp.ndarray, Ky, Kx, H, W) -> jnp.ndarray:
    """MHD [u_x, u_y, B_x, B_y]: Leray project both velocity and B-field."""
    ux, uy = x[0], x[1]
    Bx, By = x[2], x[3]
    pu_x, pu_y = _leray(ux, uy, Ky, Kx, H, W)
    pB_x, pB_y = _leray(Bx, By, Ky, Kx, H, W)
    return jnp.stack([pu_x, pu_y, pB_x, pB_y])


def _proj_sw(x: jnp.ndarray) -> jnp.ndarray:
    """SW [η, m_x, m_y]: clip height to be non-negative."""
    eta = jnp.maximum(x[0], 0.0)
    return jnp.stack([eta, x[1], x[2]])


def _proj_multiphase(x: jnp.ndarray) -> jnp.ndarray:
    """Multiphase [P, S_w, S_o]: project saturations to probability simplex.
    Exact projection: clip to [0,1] then renormalise to sum=1.
    """
    P  = x[0]
    Sw = jnp.clip(x[1], 0.0, 1.0)
    So = jnp.clip(x[2], 0.0, 1.0)
    total = Sw + So + 1e-8
    Sw = Sw / total
    So = So / total
    return jnp.stack([P, Sw, So])


def _proj_euler(x: jnp.ndarray, gamma: float = 1.4) -> jnp.ndarray:
    """Euler [ρ, m_x, m_y, E]: enforce ρ>0 and p>0.

    Minimal adjustment: only modify E where p < 0 to bring it to EPS_P.
    Keep ρ and m unchanged.
    """
    EPS_RHO = 1e-4
    EPS_P   = 1e-6

    rho = jnp.maximum(x[0], EPS_RHO)
    mx, my, E = x[1], x[2], x[3]

    ke   = (mx**2 + my**2) / (2.0 * rho)   # kinetic energy
    p    = (gamma - 1.0) * (E - ke)

    # Minimum internal energy needed for p >= EPS_P
    e_int_min = EPS_P / (gamma - 1.0)
    e_int_cur = E - ke

    # Only increase E (never decrease) to maintain p >= EPS_P
    e_int_adj = jnp.maximum(e_int_cur, e_int_min)
    E_adj     = ke + e_int_adj

    return jnp.stack([rho, mx, my, E_adj])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_projection_fn(system: str, H: int = 64, W: int = 64,
                        gamma: float = 1.4):
    """Return a JAX function  project(x) -> x_projected  for the given system.
    Works on a SINGLE sample (C, H, W) — use jax.vmap for batches.
    """
    if system in ("navier_stokes_2d",):
        Ky, Kx = _make_kgrids(H, W)
        return functools.partial(_proj_ns,  Ky=Ky, Kx=Kx, H=H, W=W)

    elif system == "mhd_2d":
        Ky, Kx = _make_kgrids(H, W)
        return functools.partial(_proj_mhd, Ky=Ky, Kx=Kx, H=H, W=W)

    elif system == "shallow_water_2d":
        return _proj_sw

    elif system == "multiphase_2d":
        return _proj_multiphase

    elif system == "euler_2d":
        return functools.partial(_proj_euler, gamma=gamma)

    else:
        raise ValueError(f"Unknown system for PCFM projection: {system}")


# Registry for quick access
PROJECT_FNS = {s: make_projection_fn(s) for s in
               ["navier_stokes_2d", "mhd_2d", "shallow_water_2d",
                "multiphase_2d", "euler_2d"]}


# ---------------------------------------------------------------------------
# PCFM Euler integration
# ---------------------------------------------------------------------------

def make_pcfm_euler_fn(net, params, system: str,
                        n_steps: int = 10, H: int = 64, W: int = 64,
                        gamma: float = 1.4):
    """
    Returns a batched, jit-compiled PCFM inference function.

    net     : FlowMap (pretrained linear baseline)
    params  : model parameters (frozen)
    system  : one of the 5 PDE system names
    n_steps : number of Euler integration steps

    Returns: fn(x0) -> xT   where x0, xT have shape (N, C, H, W)

    At each step:
        v    = v_theta(x_t, t, t)          # unconstrained vector field
        x'   = x_t + dt * v               # Euler step
        x_t+1 = project(x')               # hard constraint projection
    """
    proj_fn = make_projection_fn(system, H, W, gamma)
    ts      = jnp.linspace(0.0, 1.0, n_steps + 1)

    def single_step(x: jnp.ndarray, t_pair: jnp.ndarray) -> jnp.ndarray:
        t, dt = t_pair[0], t_pair[1]
        # unconstrained velocity from pretrained model
        v = net.apply(params, t, x, None, train=False, method="calc_b")
        # Euler step
        x_new = x + dt * v
        # project to constraint manifold
        x_new = proj_fn(x_new)
        return x_new, None

    @jax.jit
    def single_trajectory(x0: jnp.ndarray) -> jnp.ndarray:
        """x0: (C, H, W) -> xT: (C, H, W)"""
        t_pairs = jnp.stack([ts[:-1], jnp.diff(ts)], axis=1)  # (n_steps, 2)
        xT, _  = jax.lax.scan(single_step, x0, t_pairs)
        return xT

    # batch over N samples
    return jax.jit(jax.vmap(single_trajectory))
