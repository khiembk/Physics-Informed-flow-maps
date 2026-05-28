"""
Pseudo-spectral utilities for 2D periodic domains [0, 2pi]^2.

All functions operate on NumPy arrays with batch dimension N:
  fields: (N, H, W) real
  Fourier: (N, H, W//2+1) complex  (via rfft2)

Wavenumbers follow standard NumPy convention:
  kx: 0, 1, ..., W//2          (rfftfreq * W)
  ky: 0, 1, ..., H//2, -H//2+1, ..., -1  (fftfreq * H)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Wavenumber grids
# ---------------------------------------------------------------------------

def wavenumbers_2d(H: int, W: int):
    """Return (Ky, Kx) meshgrid arrays of shape (H, W//2+1)."""
    kx = np.fft.rfftfreq(W) * W   # [0, 1, ..., W//2]
    ky = np.fft.fftfreq(H) * H    # [0, 1, ..., H//2, -H//2+1, ..., -1]
    Ky, Kx = np.meshgrid(ky, kx, indexing='ij')   # (H, W//2+1)
    return Ky.astype(np.float64), Kx.astype(np.float64)


def k_squared(H: int, W: int) -> np.ndarray:
    """Return |k|^2 of shape (H, W//2+1)."""
    Ky, Kx = wavenumbers_2d(H, W)
    return Kx**2 + Ky**2


def dealias_mask(H: int, W: int) -> np.ndarray:
    """2/3-rule anti-aliasing mask of shape (H, W//2+1)."""
    Ky, Kx = wavenumbers_2d(H, W)
    return ((np.abs(Kx) <= W // 3) & (np.abs(Ky) <= H // 3)).astype(np.float64)


# ---------------------------------------------------------------------------
# Spectral derivatives  (batch-safe: inputs (..., H, W))
# ---------------------------------------------------------------------------

def grad_x_hat(f_hat: np.ndarray, Kx: np.ndarray) -> np.ndarray:
    """Fourier-space ∂/∂x: returns i*kx * f_hat."""
    return 1j * Kx * f_hat


def grad_y_hat(f_hat: np.ndarray, Ky: np.ndarray) -> np.ndarray:
    """Fourier-space ∂/∂y: returns i*ky * f_hat."""
    return 1j * Ky * f_hat


def laplacian_hat(f_hat: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Fourier-space Laplacian: returns -|k|^2 * f_hat."""
    return -K2 * f_hat


# ---------------------------------------------------------------------------
# Poisson solver  (-Δu = f  ->  u_hat = f_hat / |k|^2)
# ---------------------------------------------------------------------------

def solve_poisson(f: np.ndarray, K2: np.ndarray, H: int, W: int) -> np.ndarray:
    """Solve -Δu = f spectrally. Returns u (real, same shape as f).
    Zero-mean constraint enforced: u_hat[...,0,0] = 0.
    f: (..., H, W)
    """
    f_hat = np.fft.rfft2(f)
    K2_safe = K2.copy()
    K2_safe[0, 0] = 1.0
    u_hat = f_hat / K2_safe
    u_hat[..., 0, 0] = 0.0
    return np.fft.irfft2(u_hat, s=(H, W))


# ---------------------------------------------------------------------------
# Stream-function / vector-potential -> velocity / field
# ---------------------------------------------------------------------------

def stream_to_velocity(psi: np.ndarray, Ky: np.ndarray, Kx: np.ndarray,
                        H: int, W: int):
    """u = curl(psi): ux = ∂_y psi, uy = -∂_x psi.
    psi: (N, H, W)  ->  ux, uy: (N, H, W)
    """
    psi_hat = np.fft.rfft2(psi)
    ux = np.fft.irfft2(1j * Ky * psi_hat, s=(H, W))
    uy = np.fft.irfft2(-1j * Kx * psi_hat, s=(H, W))
    return ux, uy


def vorticity_to_velocity(omega: np.ndarray, K2: np.ndarray,
                           Ky: np.ndarray, Kx: np.ndarray,
                           H: int, W: int):
    """Recover velocity from vorticity via streamfunction.
    omega = -Δpsi  ->  psi_hat = omega_hat / |k|^2
    """
    omega_hat = np.fft.rfft2(omega)
    K2_safe = K2.copy(); K2_safe[0, 0] = 1.0
    psi_hat = omega_hat / K2_safe
    psi_hat[..., 0, 0] = 0.0
    ux = np.fft.irfft2(1j * Ky * psi_hat, s=(H, W))
    uy = np.fft.irfft2(-1j * Kx * psi_hat, s=(H, W))
    return ux, uy


# ---------------------------------------------------------------------------
# Leray projection (project velocity to divergence-free subspace)
# ---------------------------------------------------------------------------

def leray_project_hat(vx_hat: np.ndarray, vy_hat: np.ndarray,
                       K2: np.ndarray, Ky: np.ndarray, Kx: np.ndarray):
    """Project (vx_hat, vy_hat) onto divergence-free subspace.
    P = I - k k^T / |k|^2
    """
    K2_safe = K2.copy(); K2_safe[0, 0] = 1.0
    kdotu = (Kx * vx_hat + Ky * vy_hat) / K2_safe
    px_hat = vx_hat - Kx * kdotu
    py_hat = vy_hat - Ky * kdotu
    px_hat[..., 0, 0] = 0.0
    py_hat[..., 0, 0] = 0.0
    return px_hat, py_hat


def divergence_error(vx: np.ndarray, vy: np.ndarray,
                      Ky: np.ndarray, Kx: np.ndarray,
                      H: int, W: int) -> np.ndarray:
    """||div(v)||_2 / (||grad(v)||_2 + eps) per sample. Shape: (N,)."""
    vx_hat = np.fft.rfft2(vx)
    vy_hat = np.fft.rfft2(vy)
    div_hat = 1j * Kx * vx_hat + 1j * Ky * vy_hat
    div = np.fft.irfft2(div_hat, s=(H, W))

    dvx_dx = np.fft.irfft2(1j * Kx * vx_hat, s=(H, W))
    dvx_dy = np.fft.irfft2(1j * Ky * vx_hat, s=(H, W))
    dvy_dx = np.fft.irfft2(1j * Kx * vy_hat, s=(H, W))
    dvy_dy = np.fft.irfft2(1j * Ky * vy_hat, s=(H, W))
    grad_norm = np.sqrt((dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2)
                         .mean(axis=(-2, -1)))
    div_norm = np.sqrt((div**2).mean(axis=(-2, -1)))
    return div_norm / (grad_norm + 1e-12)


# ---------------------------------------------------------------------------
# Gaussian random field
# ---------------------------------------------------------------------------

def grf(N: int, H: int, W: int, alpha: float = 4.0,
        rng: np.random.Generator = None) -> np.ndarray:
    """Batch of N Gaussian random fields with power spectrum P(k) ~ |k|^{-alpha}.

    Returns real array of shape (N, H, W) with unit standard deviation per sample.
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.standard_normal((N, H, W))
    noise_hat = np.fft.rfft2(noise)   # (N, H, W//2+1)

    Ky, Kx = wavenumbers_2d(H, W)
    K = np.sqrt(Kx**2 + Ky**2)       # (H, W//2+1)
    K[0, 0] = 1.0

    weight = K ** (-alpha / 2.0)
    weight[0, 0] = 0.0  # zero mean

    field = np.fft.irfft2(noise_hat * weight[None], s=(H, W))
    std = field.std(axis=(-2, -1), keepdims=True)
    return field / (std + 1e-10)


# ---------------------------------------------------------------------------
# Constraint metrics (for validation)
# ---------------------------------------------------------------------------

def mass_error(field_T: np.ndarray, field_0: np.ndarray) -> np.ndarray:
    """|(mean(field_T) - mean(field_0))| / (|mean(field_0)| + eps). Shape: (N,)."""
    m0 = field_0.mean(axis=(-2, -1))
    mT = field_T.mean(axis=(-2, -1))
    return np.abs(mT - m0) / (np.abs(m0) + 1e-12)


def neg_violation(field: np.ndarray) -> np.ndarray:
    """mean(relu(-field)) per sample. Shape: (N,)."""
    return np.maximum(-field, 0.0).mean(axis=(-2, -1))


def simplex_error(Sw: np.ndarray, So: np.ndarray) -> np.ndarray:
    """mean(|Sw + So - 1|) per sample. Shape: (N,)."""
    return np.abs(Sw + So - 1.0).mean(axis=(-2, -1))
