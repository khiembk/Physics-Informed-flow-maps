"""
Phase 1 loss: physics residual + spatial roughness regularization.

Total loss:
    L_path(phi) = L_phy(phi) + lambda_sm * L_sm(phi)

where:
    L_phy = E_{t,x0,x1} [ w(t) * ||v_t - rhs(x_t)||^2 ]
    L_sm  = E_{t,x0,x1} [ ||grad_spatial(v_t)||^2 ]

    x_t  = (1-t)*x0 + t*x1 + alpha(t)*phi(t,x0,x1)      (physics-informed path)
    v_t  = (x1-x0) + alpha_dot(t)*phi + alpha(t)*d_phi/dt  (path velocity)
    rhs  = PDE right-hand-side at x_t                       (what the PDE predicts)
    w(t) = w0 + w_alpha * t                                 (weight increasing to t=1)

Spatial roughness  L_sm:
    The paper defines it via the state-space Jacobian of the marginal velocity.
    For PDE spatial fields (C, H, W), we use the physical-space gradient norm
    as a proxy — penalizing spatially non-smooth velocity fields:

        L_sm = mean_c [||d_x v_t^c||^2 + ||d_y v_t^c||^2]

    This is computed exactly via spectral (FFT) derivatives.
    Optionally, a Hutchinson estimator for the state-space Jacobian is
    provided as `roughness_hutchinson`.
"""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from ml_collections import config_dict


# ---------------------------------------------------------------------------
# Spatial roughness (two implementations)
# ---------------------------------------------------------------------------

def roughness_spatial(v_t: jnp.ndarray,
                       Ky: jnp.ndarray, Kx: jnp.ndarray,
                       H: int, W: int) -> jnp.ndarray:
    """
    Spatial gradient roughness of path velocity v_t.

    L_sm = mean over channels of (||d_x v||^2 + ||d_y v||^2)

    v_t : (C, H, W)  path velocity field
    Uses FFT-based spectral derivatives — exact for periodic fields.

    WHY: penalizes high-frequency spatial oscillations in v_t.
         For PDE fields, smooth paths have smooth velocities.
    """
    v_hat = jnp.fft.rfft2(v_t)                                 # (C, H, W//2+1)
    dvdx  = jnp.fft.irfft2(1j * Kx * v_hat, s=(H, W))         # (C, H, W)
    dvdy  = jnp.fft.irfft2(1j * Ky * v_hat, s=(H, W))         # (C, H, W)
    return jnp.mean(dvdx ** 2 + dvdy ** 2)


def roughness_hutchinson(v_fn: Callable, x0: jnp.ndarray, x1: jnp.ndarray,
                          t: float, rng: jnp.ndarray,
                          n_probes: int = 1) -> jnp.ndarray:
    """
    Hutchinson estimator for the state-space Jacobian norm ||J_{x0,x1} v_t||_F^2.

    Implements the paper's formulation: E[||∇_x u_phi(x_t,t)||^2]
    approximated as E_eps[||J^T eps||^2] for eps ~ N(0, I).

    v_fn(x0, x1) -> v_t   (path velocity as a function of the pair)
    n_probes: number of random probes (1 is cheap, 4+ is more accurate)

    WHY: measures how much the velocity changes when the state changes —
         high ||J|| means the path is sensitive/rough in state space.
    NOTE: more expensive than roughness_spatial; use for ablation/comparison.
    """
    keys = jax.random.split(rng, n_probes)

    def single_probe(key):
        eps = jax.random.normal(key, x0.shape)
        # J^T eps via vjp: d(v_t)/d(x0) * eps
        _, vjp_fn = jax.vjp(lambda x0_: v_fn(x0_, x1), x0)
        Jt_eps = vjp_fn(eps)[0]
        return jnp.sum(Jt_eps ** 2)

    return jnp.mean(jax.vmap(single_probe)(keys))


# ---------------------------------------------------------------------------
# Per-sample Phase 1 loss
# ---------------------------------------------------------------------------

def phase1_loss_single(
    phi_params: dict,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    path_model,
    constraint_fn: Callable,
    Ky: jnp.ndarray,
    Kx: jnp.ndarray,
    H: int,
    W: int,
    w0: float = 1.0,
    w_alpha: float = 1.0,
    lambda_sm: float = 0.01,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute L_phy + lambda_sm * L_sm for a single (t, x0, x1) triplet.

    L_phy = w(t) * ||constraint_fn(x_t)||^2
         = w(t) * ||R(x_t)||^2
    where R(x_t) measures the VIOLATION OF PHYSICS CONSTRAINTS at state x_t
    (not a dynamics mismatch):
      - SW:          relu(-eta_t)          height positivity
      - NS:          div(u_t)              divergence-free velocity
      - MHD:         [div(u_t), div(B_t)]  both divergence-free
      - Multiphase:  simplex + bound viol.  saturation constraints

    L_sm = ||grad_spatial(v_t)||^2   spatial smoothness of path velocity
    """
    # x_t: intermediate state on physics-informed path
    # v_t: path velocity (needed for L_sm only)
    x_t, v_t = path_model.apply(phi_params, t, x0, x1,
                                  method=path_model.velocity)

    # Physics loss: constraint violation at intermediate state
    R     = constraint_fn(x_t)
    w_t   = w0 + w_alpha * t
    L_phy = w_t * jnp.mean(R ** 2)

    # Spatial roughness of path velocity
    L_sm  = roughness_spatial(v_t, Ky, Kx, H, W)

    total = L_phy + lambda_sm * L_sm

    return total, {"L_phy": L_phy, "L_sm": L_sm, "L_total": total}


# ---------------------------------------------------------------------------
# Batched loss (vmap over N)
# ---------------------------------------------------------------------------

def make_phase1_loss(path_model, constraint_fn, Ky, Kx, H, W,
                      w0=1.0, w_alpha=1.0, lambda_sm=0.01):
    """
    Returns a batched loss function:
        loss_fn(phi_params, x0, x1, t) -> (scalar, metrics)

    x0, x1: (N, C, H, W)
    t:      (N,)
    """
    @functools.partial(jax.vmap, in_axes=(0, 0, 0))
    def per_sample(x0_i, x1_i, t_i):
        return phase1_loss_single(
            phi_params, x0_i, x1_i, t_i,
            path_model, rhs_fn, Ky, Kx, H, W,
            w0, w_alpha, lambda_sm,
        )

    def loss_fn(phi_params_arg, x0, x1, t):
        # Closure over per_sample doesn't re-vmap each call; use jit-friendly form
        losses, metrics = jax.vmap(
            lambda x0_i, x1_i, t_i: phase1_loss_single(
                phi_params_arg, x0_i, x1_i, t_i,
                path_model, constraint_fn, Ky, Kx, H, W,
                w0, w_alpha, lambda_sm,
            )
        )(x0, x1, t)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, metrics)

    return loss_fn


# ---------------------------------------------------------------------------
# Phase1-Dynamic: combined constraint + dynamics mismatch loss
# ---------------------------------------------------------------------------

def phase1_dynamic_loss_single(
    phi_params: dict,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray,
    path_model,
    constraint_fn: Callable,
    rhs_fn: Callable,
    Ky: jnp.ndarray,
    Kx: jnp.ndarray,
    H: int,
    W: int,
    w0: float = 1.0,
    w_alpha: float = 1.0,
    lambda_sm: float = 0.0,
    alpha_dyn: float = 1.0,
) -> Tuple[jnp.ndarray, dict]:
    """
    Combined Phase 1 loss: constraint violation + dynamics mismatch.

    L_phy = w(t) * ( ||R_con(x_t)||²  +  alpha_dyn * ||v_t - rhs(x_t)||² )

    R_con = constraint_fn(x_t)  — algebraic constraint violation
    R_dyn = v_t - rhs_fn(x_t)  — path velocity vs PDE dynamics

    Why combine:
    - R_con ≈ 0 when phi≈0 (dead gradient problem)
    - R_dyn is LARGE from step 1 (linear velocity ≠ PDE velocity)
    - R_dyn provides initial gradient; R_con fine-tunes constraint satisfaction
    - PDE dynamics implicitly preserve constraints (e.g. NS rhs → div-free)
    """
    # Path state and velocity
    x_t, v_t = path_model.apply(phi_params, t, x0, x1,
                                  method=path_model.velocity)

    # Constraint violation (small when phi≈0)
    R_con = constraint_fn(x_t)
    L_con = jnp.mean(R_con ** 2)

    # Dynamics mismatch (large from step 1)
    rhs   = rhs_fn(x_t)
    R_dyn = v_t - rhs
    L_dyn = jnp.mean(R_dyn ** 2)

    w_t   = w0 + w_alpha * t
    L_phy = w_t * (L_con + alpha_dyn * L_dyn)

    # Spatial roughness
    L_sm  = roughness_spatial(v_t, Ky, Kx, H, W)

    total = L_phy + lambda_sm * L_sm

    return total, {
        "L_phy":   L_phy,
        "L_con":   L_con,
        "L_dyn":   L_dyn,
        "L_sm":    L_sm,
        "L_total": total,
    }


def make_phase1_dynamic_loss(path_model, constraint_fn, rhs_fn,
                               Ky, Kx, H, W,
                               w0=1.0, w_alpha=1.0,
                               lambda_sm=0.0, alpha_dyn=1.0):
    """Batched combined loss. Returns loss_fn(phi_params, x0, x1, t)."""
    def loss_fn(phi_params_arg, x0, x1, t):
        losses, metrics = jax.vmap(
            lambda x0_i, x1_i, t_i: phase1_dynamic_loss_single(
                phi_params_arg, x0_i, x1_i, t_i,
                path_model, constraint_fn, rhs_fn,
                Ky, Kx, H, W, w0, w_alpha, lambda_sm, alpha_dyn,
            )
        )(x0, x1, t)
        return jnp.mean(losses), jax.tree_util.tree_map(jnp.mean, metrics)
    return loss_fn


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnums=(3, 4))
def train_step(phi_params, opt_state, batch, loss_fn, optimizer):
    """One gradient step on Phase 1 loss.

    batch: dict with keys 'x0', 'xT', 't'
    loss_fn: from make_phase1_loss(...)
    """
    def _loss(params):
        return loss_fn(params, batch['x0'], batch['xT'], batch['t'])

    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(phi_params)
    updates, opt_state_new = optimizer.update(grads, opt_state, phi_params)
    phi_params_new = optax.apply_updates(phi_params, updates)
    return phi_params_new, opt_state_new, loss, metrics
