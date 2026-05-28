"""
Pseudo-spectral Navier-Stokes / Boussinesq solver on [0, 2pi]^2.

PDE (vorticity-streamfunction + scalar transport):
    d_t omega + u . grad omega = nu * Delta omega + d_x c   (buoyancy source)
    d_t c     + u . grad c     = kappa * Delta c

    omega = -Delta psi,  u = (d_y psi, -d_x psi)

State stored: [c, u_x, u_y]   shape (N, 3, H, W)
Constraint:   div(u_T) = 0
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.spectral import (
    wavenumbers_2d, k_squared, dealias_mask,
    grad_x_hat, grad_y_hat, laplacian_hat,
    vorticity_to_velocity, grf, divergence_error,
)


# ---------------------------------------------------------------------------
# RHS in Fourier space
# ---------------------------------------------------------------------------

def _rhs(omega_hat, c_hat, K2, Ky, Kx, mask, H, W, nu, kappa):
    """Returns (d_omega_hat/dt, d_c_hat/dt)."""
    # Dealiased fields
    oh = omega_hat * mask
    ch = c_hat * mask

    # Streamfunction -> velocity
    K2s = K2.copy(); K2s[0, 0] = 1.0
    psi_hat = oh / K2s
    psi_hat[..., 0, 0] = 0.0
    ux = np.fft.irfft2(1j * Ky * psi_hat, s=(H, W))
    uy = np.fft.irfft2(-1j * Kx * psi_hat, s=(H, W))

    # Physical-space gradients (for nonlinear terms)
    domega_dx = np.fft.irfft2(1j * Kx * oh, s=(H, W))
    domega_dy = np.fft.irfft2(1j * Ky * oh, s=(H, W))
    dc_dx     = np.fft.irfft2(1j * Kx * ch, s=(H, W))
    dc_dy     = np.fft.irfft2(1j * Ky * ch, s=(H, W))

    # Nonlinear advection -> back to Fourier, dealias
    adv_omega = np.fft.rfft2(ux * domega_dx + uy * domega_dy) * mask
    adv_c     = np.fft.rfft2(ux * dc_dx     + uy * dc_dy    ) * mask

    # Buoyancy source: d_x c (acts on omega equation)
    buoy = 1j * Kx * ch

    # Full RHS
    d_omega = (-adv_omega + nu * laplacian_hat(oh, K2) + buoy) * mask
    d_c     = (-adv_c     + kappa * laplacian_hat(ch, K2)    ) * mask
    return d_omega, d_c


def _rk4(omega_hat, c_hat, dt, K2, Ky, Kx, mask, H, W, nu, kappa):
    """One RK4 step."""
    def rhs(oh, ch):
        return _rhs(oh, ch, K2, Ky, Kx, mask, H, W, nu, kappa)

    k1o, k1c = rhs(omega_hat, c_hat)
    k2o, k2c = rhs(omega_hat + 0.5*dt*k1o, c_hat + 0.5*dt*k1c)
    k3o, k3c = rhs(omega_hat + 0.5*dt*k2o, c_hat + 0.5*dt*k2c)
    k4o, k4c = rhs(omega_hat +     dt*k3o, c_hat +     dt*k3c)

    new_o = (omega_hat + (dt/6) * (k1o + 2*k2o + 2*k3o + k4o)) * mask
    new_c = (c_hat     + (dt/6) * (k1c + 2*k2c + 2*k3c + k4c)) * mask
    return new_o, new_c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(cfg: dict, seed: int) -> dict:
    """Generate paired (x0, xT) samples for NS/Boussinesq.

    cfg keys: N, H, W, nu, kappa, T, dt, omega0_scale, c0_scale, grf_alpha
    Returns dict with keys: x0, xT, t0, tT, dt, channel_names, params
    """
    N      = cfg["N"]
    H      = cfg.get("H", 64)
    W      = cfg.get("W", 64)
    nu     = cfg.get("nu", 0.001)
    kappa  = cfg.get("kappa", 0.0005)
    T      = cfg.get("T", 0.1)
    dt     = cfg.get("dt", 0.001)
    omega0_scale = cfg.get("omega0_scale", 1.0)
    c0_scale     = cfg.get("c0_scale", 1.0)
    grf_alpha    = cfg.get("grf_alpha", 4.0)

    rng = np.random.default_rng(seed)
    Ky, Kx = wavenumbers_2d(H, W)
    K2   = k_squared(H, W)
    mask = dealias_mask(H, W)

    # Initial conditions
    omega0 = omega0_scale * grf(N, H, W, alpha=grf_alpha, rng=rng)
    c0     = c0_scale     * grf(N, H, W, alpha=grf_alpha - 1.0, rng=rng)

    omega_hat = np.fft.rfft2(omega0) * mask
    c_hat     = np.fft.rfft2(c0)     * mask

    # Recover initial velocity for x0
    K2s = K2.copy(); K2s[0, 0] = 1.0
    psi0_hat = omega_hat / K2s; psi0_hat[..., 0, 0] = 0.0
    ux0 = np.fft.irfft2(1j * Ky * psi0_hat, s=(H, W))
    uy0 = np.fft.irfft2(-1j * Kx * psi0_hat, s=(H, W))

    x0 = np.stack([c0, ux0, uy0], axis=1)   # (N, 3, H, W)

    # Time integration
    n_steps = int(round(T / dt))
    for step in range(n_steps):
        omega_hat, c_hat = _rk4(omega_hat, c_hat, dt,
                                  K2, Ky, Kx, mask, H, W, nu, kappa)

    # Final state
    omega_T = np.fft.irfft2(omega_hat, s=(H, W))
    c_T     = np.fft.irfft2(c_hat,     s=(H, W))
    K2s = K2.copy(); K2s[0, 0] = 1.0
    psiT_hat = omega_hat / K2s; psiT_hat[..., 0, 0] = 0.0
    uxT = np.fft.irfft2(1j * Ky * psiT_hat, s=(H, W))
    uyT = np.fft.irfft2(-1j * Kx * psiT_hat, s=(H, W))

    xT = np.stack([c_T, uxT, uyT], axis=1)   # (N, 3, H, W)

    # Constraint check
    div_err = divergence_error(uxT, uyT, Ky, Kx, H, W)
    print(f"  [NS] DivErr(u_T): mean={div_err.mean():.2e}  max={div_err.max():.2e}")

    return {
        "x0": x0.astype(np.float32),
        "xT": xT.astype(np.float32),
        "t0": 0.0, "tT": float(T), "dt": float(dt),
        "channel_names": ["c", "u_x", "u_y"],
        "params": {"nu": nu, "kappa": kappa, "H": H, "W": W},
    }
