"""
Code for initializing datasets.
"""

import functools
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from ml_collections import config_dict


def sample_checkerboard(
    n_samples: int, key: jnp.ndarray, *, n_squares: int
) -> np.ndarray:
    """
    Samples the checkerboard dataset on [-1,1] x [-1,1]
    with alternating squares removed.
    """
    del key
    total_samples = 0
    samples = np.array([]).reshape((0, 2))

    while total_samples < n_samples:
        curr_samples = np.random.rand(n_samples * 2, 2)

        x_idx = (curr_samples[:, 0] * n_squares).astype(int)
        y_idx = (curr_samples[:, 1] * n_squares).astype(int)

        mask = (x_idx + y_idx) % 2 == 0
        curr_samples = curr_samples[mask]

        samples = np.concatenate((samples, curr_samples))
        total_samples = samples.shape[0]

    return 2 * samples[:n_samples] - 1


def setup_base(cfg: config_dict.ConfigDict, ex_input: jnp.ndarray) -> Callable:
    """Set up the base density for the system."""
    if cfg.problem.base == "gaussian":

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho0(bs: int, key: jnp.ndarray):
            return cfg.network.rescale * jax.random.normal(
                key, shape=(bs, *ex_input.shape)
            )

    else:
        raise ValueError("Specified base density is not implemented.")

    return sample_rho0


def np_to_tfds(cfg: config_dict.ConfigDict, x1s: np.ndarray) -> tf.data.Dataset:
    """Given a NumPy array, convert to a TensorFlow dataset with batching and shuffling."""
    return (
        tf.data.Dataset.from_tensor_slices(x1s)
        .shuffle(50_000, reshuffle_each_iteration=True)
        .repeat()
        .batch(cfg.optimization.bs)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )


def setup_target(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray):
    """Set up the target density for the system."""
    if cfg.problem.target == "checker":
        assert cfg.problem.d == 2, "Checkerboard only implemented for d=2."

        @functools.partial(jax.jit, static_argnums=(0,))
        def sample_rho1(num_samples: int, key: jnp.ndarray) -> jnp.ndarray:
            return sample_checkerboard(num_samples, key, n_squares=4)

        n_samples = cfg.problem.n
        key, prng_key = jax.random.split(prng_key)
        x1s = sample_rho1(n_samples, key)
        rescale_value = float(np.std(x1s))
        ds = np_to_tfds(cfg, x1s)

    else:
        raise ValueError(f"Unknown target density: {cfg.problem.target!r}. "
                         "Add a custom dataset loader here.")

    if cfg.problem.gaussian_scale == "adaptive":
        cfg.network.rescale = rescale_value
    else:
        cfg.network.rescale = 1.0

    return cfg, ds, prng_key
