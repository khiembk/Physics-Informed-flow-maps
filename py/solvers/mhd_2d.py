"""
Pseudo-spectral incompressible MHD solver on [0, 2pi]^2.

Uses stream function psi and vector potential A:
    u = (d_y psi, -d_x psi)   => div(u) = 0
    B = (d_y A,   -d_x A  )   => div(B) = 0

Vorticity omega = -Delta psi,  current j = -Delta A.

Evolution equations:
    d_t omega = J(A, j) - J(psi, omega) + nu * Delta omega
    d_t A     = -J(psi, A)              + eta * Delta A

    J(f, g) = d_x f * d_y g - d_y f * d_x g  (2D Jacobian / Poisson bracket)

State stored: [u_x, u_y, B_x, B_y]   shape (N, 4, H, W)
Constraints:  div(u_T) = 0,  div(B_T) = 0
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.spectral import (
    wavenumbers_2d, k_squared, dealias_mask,
    laplacian_hat, grf, divergence_error,
)


# ---------------------------------------------------------------------------
# 2D Jacobian J(f, g) = d_x f * d_y g - d_y f * d_x g
# ---------------------------------------------------------------------------

def _jacobian(f_hat, g_hat, Ky, Kx, mask, H, W):
    """Dealiased Jacobian J(f,g) returned as Fourier coefficients."""
    fx = np.fft.irfft2(1j * Kx * f_hat * mask, s=(H, W))
    fy = np.fft.irfft2(1j * Ky * f_hat * mask, s=(H, W))
    gx = np.fft.irfft2(1j * Kx * g_hat * mask, s=(H, W))
    gy = np.fft.irfft2(1j * Ky * g_hat * mask, s=(H, W))
    return np.fft.rfft2(fx * gy - fy * gx) * mask


def _hat_to_psi(omega_hat, K2):
    K2s = K2.copy(); K2s[0, 0] = 1.0
    psi_hat = omega_hat / K2s
    psi_hat[..., 0, 0] = 0.0
    return psi_hat


def _rhs(omega_hat, A_hat, K2, Ky, Kx, mask, H, W, nu, eta):
    oh = omega_hat * mask
    Ah = A_hat     * mask

    psi_hat = _hat_to_psi(oh, K2)
    j_hat   = -laplacian_hat(Ah, K2) * mask   # j = -Delta A  (in Fourier: |k|^2 * A_hat)

    # Nonlinear terms
    J_A_j   = _jacobian(Ah,      j_hat,   Ky, Kx, mask, H, W)
    J_psi_w = _jacobian(psi_hat, oh,      Ky, Kx, mask, H, W)
    J_psi_A = _jacobian(psi_hat, Ah,      Ky, Kx, mask, H, W)

    d_omega = (J_A_j - J_psi_w + nu  * laplacian_hat(oh, K2)) * mask
    d_A     = (-J_psi_A         + eta * laplacian_hat(Ah, K2)) * mask
    return d_omega, d_A


def _rk4(omega_hat, A_hat, dt, K2, Ky, Kx, mask, H, W, nu, eta):
    def rhs(oh, Ah):
        return _rhs(oh, Ah, K2, Ky, Kx, mask, H, W, nu, eta)

    k1o, k1A = rhs(omega_hat, A_hat)
    k2o, k2A = rhs(omega_hat + 0.5*dt*k1o, A_hat + 0.5*dt*k1A)
    k3o, k3A = rhs(omega_hat + 0.5*dt*k2o, A_hat + 0.5*dt*k2A)
    k4o, k4A = rhs(omega_hat +     dt*k3o, A_hat +     dt*k3A)

    new_o = (omega_hat + (dt/6) * (k1o + 2*k2o + 2*k3o + k4o)) * mask
    new_A = (A_hat     + (dt/6) * (k1A + 2*k2A + 2*k3A + k4A)) * mask
    return new_o, new_A


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(cfg: dict, seed: int) -> dict:
    N    = cfg["N"]
    H    = cfg.get("H", 64)
    W    = cfg.get("W", 64)
    nu   = cfg.get("nu",  0.001)
    eta  = cfg.get("eta", 0.001)
    T    = cfg.get("T",   0.05)
    dt   = cfg.get("dt",  0.0005)
    grf_alpha = cfg.get("grf_alpha", 4.0)
    amp  = cfg.get("amplitude", 1.0)

    rng = np.random.default_rng(seed)
    Ky, Kx = wavenumbers_2d(H, W)
    K2   = k_squared(H, W)
    mask = dealias_mask(H, W)

    # Initial conditions: sample omega_0 and A_0 from GRF
    omega0 = amp * grf(N, H, W, alpha=grf_alpha, rng=rng)
    A0     = amp * grf(N, H, W, alpha=grf_alpha, rng=rng)

    omega_hat = np.fft.rfft2(omega0) * mask
    A_hat     = np.fft.rfft2(A0)     * mask

    # Build initial state [ux, uy, Bx, By]
    def _uB_from_hats(oh, Ah):
        psi_h = _hat_to_psi(oh, K2)
        ux = np.fft.irfft2(1j * Ky * psi_h, s=(H, W))
        uy = np.fft.irfft2(-1j * Kx * psi_h, s=(H, W))
        Bx = np.fft.irfft2(1j * Ky * Ah, s=(H, W))
        By = np.fft.irfft2(-1j * Kx * Ah, s=(H, W))
        return ux, uy, Bx, By

    ux0, uy0, Bx0, By0 = _uB_from_hats(omega_hat, A_hat)
    x0 = np.stack([ux0, uy0, Bx0, By0], axis=1)   # (N, 4, H, W)

    # Time integration
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        omega_hat, A_hat = _rk4(omega_hat, A_hat, dt,
                                  K2, Ky, Kx, mask, H, W, nu, eta)

    uxT, uyT, BxT, ByT = _uB_from_hats(omega_hat, A_hat)
    xT = np.stack([uxT, uyT, BxT, ByT], axis=1)   # (N, 4, H, W)

    # Constraint checks
    divu = divergence_error(uxT, uyT, Ky, Kx, H, W)
    divB = divergence_error(BxT, ByT, Ky, Kx, H, W)
    print(f"  [MHD] DivErr(u_T): {divu.mean():.2e}  DivErr(B_T): {divB.mean():.2e}")

    return {
        "x0": x0.astype(np.float32),
        "xT": xT.astype(np.float32),
        "t0": 0.0, "tT": float(T), "dt": float(dt),
        "channel_names": ["u_x", "u_y", "B_x", "B_y"],
        "params": {"nu": nu, "eta": eta, "H": H, "W": W},
    }
