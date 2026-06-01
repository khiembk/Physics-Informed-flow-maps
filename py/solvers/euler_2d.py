"""
2D compressible Euler equations with artificial viscosity on [0, 2pi]^2.

Conservative form:
    d_t rho + div(m)              = nu * Delta rho
    d_t m   + div(m x u + p * I) = nu * Delta u
    d_t E   + div((E+p) * u)     = nu * Delta (E/rho)

Ideal gas EOS:  p = (gamma-1) * (E - |m|^2 / (2*rho))

State stored: [rho, m_x, m_y, E]   shape (N, 4, H, W)

KEY NONLINEAR CONSTRAINTS (violated by linear interpolation):
    rho > 0   (density positivity)
    p   > 0   (pressure positivity)

Why linear interpolation violates p > 0:
    p_t = (gamma-1)(E_t - |m_t|^2 / (2*rho_t))
    E_t, m_t, rho_t are linear combos of endpoints,
    but |m_t|^2 / (2*rho_t) is NONLINEAR -> p_t can go negative
    even when p_0 > 0 and p_T > 0.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.spectral import (
    wavenumbers_2d, k_squared, dealias_mask,
    laplacian_hat, grf, neg_violation,
)


EPS_RHO = 1e-4    # floor density to prevent div/0
EPS_P   = 1e-6    # floor pressure


def _pressure(rho, mx, my, E, gamma):
    """p = (gamma-1)(E - |m|^2/(2*rho))."""
    rho_s = np.maximum(rho, EPS_RHO)
    return (gamma - 1.0) * (E - (mx**2 + my**2) / (2.0 * rho_s))


def _rhs(rho_hat, mx_hat, my_hat, E_hat,
         K2, Ky, Kx, mask, H, W, gamma, nu):
    """RHS of compressible Euler + artificial viscosity."""

    def irfft(h): return np.fft.irfft2(h * mask, s=(H, W))
    def rfft(f):  return np.fft.rfft2(f) * mask
    def div_hat(Fx, Fy):
        return 1j * Kx * rfft(Fx) + 1j * Ky * rfft(Fy)

    # Physical fields
    rho = irfft(rho_hat)
    mx  = irfft(mx_hat)
    my  = irfft(my_hat)
    E   = irfft(E_hat)

    rho_s = np.maximum(rho, EPS_RHO)
    ux    = mx / rho_s
    uy    = my / rho_s
    p     = _pressure(rho, mx, my, E, gamma)
    p     = np.maximum(p, EPS_P)    # floor in solver (not constraint)

    # Continuity: d_t rho = -div(m) + nu*Lap(rho)
    d_rho_hat = -div_hat(mx, my) + nu * laplacian_hat(rho_hat, K2)

    # Momentum: d_t m = -div(m x u + p I) + nu*Lap(u)
    d_mx_hat  = (-div_hat(mx * ux + p, mx * uy)
                 + nu * laplacian_hat(np.fft.rfft2(ux) * mask, K2))
    d_my_hat  = (-div_hat(my * ux, my * uy + p)
                 + nu * laplacian_hat(np.fft.rfft2(uy) * mask, K2))

    # Energy: d_t E = -div((E+p)*u) + nu*Lap(E/rho)
    d_E_hat   = (-div_hat((E + p) * ux, (E + p) * uy)
                 + nu * laplacian_hat(np.fft.rfft2(E / rho_s) * mask, K2))

    return (d_rho_hat * mask, d_mx_hat * mask,
            d_my_hat * mask, d_E_hat  * mask)


def _rk4(rho_hat, mx_hat, my_hat, E_hat, dt, K2, Ky, Kx, mask, H, W, gamma, nu):
    def rhs(rh, mxh, myh, Eh):
        return _rhs(rh, mxh, myh, Eh, K2, Ky, Kx, mask, H, W, gamma, nu)

    k1 = rhs(rho_hat, mx_hat, my_hat, E_hat)
    k2 = rhs(rho_hat+.5*dt*k1[0], mx_hat+.5*dt*k1[1],
             my_hat +.5*dt*k1[2], E_hat  +.5*dt*k1[3])
    k3 = rhs(rho_hat+.5*dt*k2[0], mx_hat+.5*dt*k2[1],
             my_hat +.5*dt*k2[2], E_hat  +.5*dt*k2[3])
    k4 = rhs(rho_hat+   dt*k3[0], mx_hat+   dt*k3[1],
             my_hat +   dt*k3[2], E_hat  +   dt*k3[3])

    c = dt / 6.0
    new = [
        (rho_hat + c*(k1[0]+2*k2[0]+2*k3[0]+k4[0])) * mask,
        (mx_hat  + c*(k1[1]+2*k2[1]+2*k3[1]+k4[1])) * mask,
        (my_hat  + c*(k1[2]+2*k2[2]+2*k3[2]+k4[2])) * mask,
        (E_hat   + c*(k1[3]+2*k2[3]+2*k3[3]+k4[3])) * mask,
    ]
    return tuple(new)


def generate(cfg: dict, seed: int) -> dict:
    """
    Generate paired (x0, xT) for compressible Euler 2D.

    cfg keys: N, H, W, gamma, nu, T, dt,
              rho_mean, rho_amp, mach, grf_alpha
    """
    N         = cfg['N']
    H         = cfg.get('H', 64)
    W         = cfg.get('W', 64)
    gamma     = cfg.get('gamma', 1.4)
    nu        = cfg.get('nu', 0.005)
    T         = cfg.get('T', 0.3)
    dt        = cfg.get('dt', 0.001)
    rho_mean  = cfg.get('rho_mean', 1.0)
    rho_amp   = cfg.get('rho_amp', 0.3)
    mach      = cfg.get('mach', 0.3)
    grf_alpha = cfg.get('grf_alpha', 4.0)

    rng = np.random.default_rng(seed)
    Ky, Kx = wavenumbers_2d(H, W)
    K2      = k_squared(H, W)
    mask    = dealias_mask(H, W)

    cs = 1.0   # sound speed (normalized)

    # Initial conditions
    rho0 = np.clip(rho_mean + rho_amp * grf(N, H, W, alpha=grf_alpha, rng=rng),
                   0.2, 3.0)                                      # (N, H, W)
    ux0  = mach * cs * grf(N, H, W, alpha=grf_alpha - 1.0, rng=rng)
    uy0  = mach * cs * grf(N, H, W, alpha=grf_alpha - 1.0, rng=rng)
    mx0  = rho0 * ux0
    my0  = rho0 * uy0

    # Pressure from isentropic init: p0 = cs^2 * rho^gamma
    p0   = cs**2 * rho0**gamma
    E0   = p0 / (gamma - 1.0) + (mx0**2 + my0**2) / (2.0 * rho0)

    x0 = np.stack([rho0, mx0, my0, E0], axis=1)   # (N, 4, H, W)

    rho_hat = np.fft.rfft2(rho0) * mask
    mx_hat  = np.fft.rfft2(mx0)  * mask
    my_hat  = np.fft.rfft2(my0)  * mask
    E_hat   = np.fft.rfft2(E0)   * mask

    # Time integration
    n_steps = int(round(T / dt))
    for _ in range(n_steps):
        rho_hat, mx_hat, my_hat, E_hat = _rk4(
            rho_hat, mx_hat, my_hat, E_hat, dt,
            K2, Ky, Kx, mask, H, W, gamma, nu
        )

    rhoT = np.fft.irfft2(rho_hat, s=(H, W))
    mxT  = np.fft.irfft2(mx_hat,  s=(H, W))
    myT  = np.fft.irfft2(my_hat,  s=(H, W))
    ET   = np.fft.irfft2(E_hat,   s=(H, W))
    pT   = _pressure(rhoT, mxT, myT, ET, gamma)

    # Check constraints at ENDPOINTS
    neg_rho = float(np.maximum(-rhoT, 0).mean())
    neg_p   = float(np.maximum(-pT,   0).mean())
    print(f"  [Euler] NegRho(T): {neg_rho:.2e}  NegP(T): {neg_p:.2e}")

    # Check pressure violation on LINEAR INTERPOLANT at t=0.5
    rho_mid = 0.5 * rho0 + 0.5 * rhoT
    mx_mid  = 0.5 * mx0  + 0.5 * mxT
    my_mid  = 0.5 * my0  + 0.5 * myT
    E_mid   = 0.5 * E0   + 0.5 * ET
    p_mid   = _pressure(rho_mid, mx_mid, my_mid, E_mid, gamma)
    neg_p_lin = float(np.maximum(-p_mid, 0).mean())
    frac_neg  = float((p_mid < 0).mean())
    print(f"  [Euler] Linear interp t=0.5: NegP={neg_p_lin:.2e}  "
          f"FracNeg={frac_neg:.4f}  (nonzero = constraint violated!)")

    xT = np.stack([rhoT, mxT, myT, ET], axis=1)   # (N, 4, H, W)

    return {
        'x0': x0.astype(np.float32),
        'xT': xT.astype(np.float32),
        't0': 0.0, 'tT': float(T), 'dt': float(dt),
        'channel_names': ['rho', 'm_x', 'm_y', 'E'],
        'params': {'gamma': gamma, 'nu': nu, 'H': H, 'W': W},
    }
