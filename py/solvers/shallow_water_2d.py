"""
Pseudo-spectral shallow water equations on [0, 2pi]^2.

Conservative form:
    d_t eta + d_x mx + d_y my = 0
    d_t mx  + d_x(mx^2/eta + g*eta^2/2) + d_y(mx*my/eta) = nu * Delta ux
    d_t my  + d_x(mx*my/eta) + d_y(my^2/eta + g*eta^2/2) = nu * Delta uy

where ux = mx/eta, uy = my/eta.

Diffusion applied spectrally; nonlinear fluxes in physical space.

State stored: [eta, m_x, m_y]   shape (N, 3, H, W)
Constraints:  eta_T >= 0,  sum(eta_T) ≈ sum(eta_0)  (mass conservation)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.spectral import (
    wavenumbers_2d, k_squared, dealias_mask,
    laplacian_hat, grf, neg_violation, mass_error,
)

ETA_MIN = 1e-3   # floor to prevent division by zero in flux computation


def _fluxes(eta, mx, my, g):
    """Conservative fluxes F = [Fx_eta, Fx_mx, Fx_my, Fy_eta, Fy_mx, Fy_my]."""
    eta_s = np.maximum(eta, ETA_MIN)
    ux = mx / eta_s
    uy = my / eta_s

    Fx_eta = mx
    Fx_mx  = mx * ux + 0.5 * g * eta**2
    Fx_my  = mx * uy

    Fy_eta = my
    Fy_mx  = my * ux
    Fy_my  = my * uy + 0.5 * g * eta**2

    return Fx_eta, Fx_mx, Fx_my, Fy_eta, Fy_mx, Fy_my


def _rhs(eta_hat, mx_hat, my_hat, K2, Ky, Kx, mask, H, W, g, nu):
    """RHS for (eta_hat, mx_hat, my_hat). Returns derivatives in Fourier space."""
    # Physical-space fields
    eta = np.fft.irfft2(eta_hat * mask, s=(H, W))
    mx  = np.fft.irfft2(mx_hat  * mask, s=(H, W))
    my  = np.fft.irfft2(my_hat  * mask, s=(H, W))

    # Conservative fluxes
    Fx_eta, Fx_mx, Fx_my, Fy_eta, Fy_mx, Fy_my = _fluxes(eta, mx, my, g)

    # Divergence via spectral derivatives
    def _div_x_y(Fx, Fy):
        return (np.fft.rfft2(Fx) * 1j * Kx + np.fft.rfft2(Fy) * 1j * Ky) * mask

    d_eta_hat = -_div_x_y(Fx_eta, Fy_eta)
    d_mx_hat  = -_div_x_y(Fx_mx,  Fy_mx)
    d_my_hat  = -_div_x_y(Fx_my,  Fy_my)

    # Viscous diffusion (applied to velocity, not momentum):
    # nu * Delta u -> in momentum: nu * Delta (m/eta) ≈ nu * Delta m / eta_mean
    # Simplified: apply viscosity as nu * Delta m directly (standard spectral SW)
    eta_s = np.maximum(eta, ETA_MIN)
    ux_hat = np.fft.rfft2(mx / eta_s) * mask
    uy_hat = np.fft.rfft2(my / eta_s) * mask

    # Viscous force in momentum: f_visc = nu * Delta u * eta
    # Approximate: nu * Delta m  (valid for near-uniform eta)
    d_mx_hat += nu * laplacian_hat(mx_hat * mask, K2)
    d_my_hat += nu * laplacian_hat(my_hat * mask, K2)

    return d_eta_hat * mask, d_mx_hat * mask, d_my_hat * mask


def _rk4(eta_hat, mx_hat, my_hat, dt, K2, Ky, Kx, mask, H, W, g, nu):
    def rhs(eh, mxh, myh):
        return _rhs(eh, mxh, myh, K2, Ky, Kx, mask, H, W, g, nu)

    k1e, k1x, k1y = rhs(eta_hat, mx_hat, my_hat)
    k2e, k2x, k2y = rhs(eta_hat+0.5*dt*k1e, mx_hat+0.5*dt*k1x, my_hat+0.5*dt*k1y)
    k3e, k3x, k3y = rhs(eta_hat+0.5*dt*k2e, mx_hat+0.5*dt*k2x, my_hat+0.5*dt*k2y)
    k4e, k4x, k4y = rhs(eta_hat+    dt*k3e, mx_hat+    dt*k3x, my_hat+    dt*k3y)

    coeff = dt / 6.0
    ne = (eta_hat + coeff*(k1e+2*k2e+2*k3e+k4e)) * mask
    nx = (mx_hat  + coeff*(k1x+2*k2x+2*k3x+k4x)) * mask
    ny = (my_hat  + coeff*(k1y+2*k2y+2*k3y+k4y)) * mask
    return ne, nx, ny


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(cfg: dict, seed: int) -> dict:
    N    = cfg["N"]
    H    = cfg.get("H", 64)
    W    = cfg.get("W", 64)
    g    = cfg.get("g",  1.0)
    nu   = cfg.get("nu", 0.002)
    T    = cfg.get("T",  0.05)
    dt   = cfg.get("dt", 0.0005)
    eta_mean  = cfg.get("eta_mean", 1.0)
    eta_amp   = cfg.get("eta_amp",  0.1)
    mom_amp   = cfg.get("mom_amp",  0.0)   # 0 = zero initial velocity
    grf_alpha = cfg.get("grf_alpha", 4.0)

    rng = np.random.default_rng(seed)
    Ky, Kx = wavenumbers_2d(H, W)
    K2   = k_squared(H, W)
    mask = dealias_mask(H, W)

    # Initial height: eta_0 = eta_mean + eta_amp * GRF, clipped to > ETA_MIN
    h_grf = grf(N, H, W, alpha=grf_alpha, rng=rng)
    eta0  = np.clip(eta_mean + eta_amp * h_grf, ETA_MIN * 2, None)

    # Initial momentum (zero or small)
    if mom_amp > 0:
        mx0 = mom_amp * eta0 * grf(N, H, W, alpha=grf_alpha, rng=rng)
        my0 = mom_amp * eta0 * grf(N, H, W, alpha=grf_alpha, rng=rng)
    else:
        mx0 = np.zeros_like(eta0)
        my0 = np.zeros_like(eta0)

    x0 = np.stack([eta0, mx0, my0], axis=1)   # (N, 3, H, W)

    eta_hat = np.fft.rfft2(eta0) * mask
    mx_hat  = np.fft.rfft2(mx0)  * mask
    my_hat  = np.fft.rfft2(my0)  * mask

    # Time integration
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        eta_hat, mx_hat, my_hat = _rk4(eta_hat, mx_hat, my_hat, dt,
                                         K2, Ky, Kx, mask, H, W, g, nu)

    etaT = np.fft.irfft2(eta_hat, s=(H, W))
    mxT  = np.fft.irfft2(mx_hat,  s=(H, W))
    myT  = np.fft.irfft2(my_hat,  s=(H, W))

    xT = np.stack([etaT, mxT, myT], axis=1)   # (N, 3, H, W)

    # Constraint checks
    neg_h  = neg_violation(etaT)
    mass_e = mass_error(etaT, eta0)
    print(f"  [SW] NegHeight: {neg_h.mean():.2e}  MassErr: {mass_e.mean():.2e}")

    return {
        "x0": x0.astype(np.float32),
        "xT": xT.astype(np.float32),
        "t0": 0.0, "tT": float(T), "dt": float(dt),
        "channel_names": ["eta", "m_x", "m_y"],
        "params": {"g": g, "nu": nu, "H": H, "W": W},
    }
