"""
Code for wandb visualization and logging.
"""

import functools
import signal
import sys
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
import wandb
from flax.serialization import to_bytes
from jax.flatten_util import ravel_pytree
from matplotlib import pyplot as plt
from ml_collections import config_dict

from . import datasets, dist_utils, flow_map, state_utils

Parameters = Dict[str, Dict]


def get_params_for_sampling(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    param_type: str = "visual",
) -> jnp.ndarray:
    """Get the appropriate parameters for sampling (visualization or evaluation)."""
    config_param = f"{param_type}_ema_factor"

    if (
        hasattr(cfg.logging, config_param)
        and getattr(cfg.logging, config_param) is not None
    ):
        ema_factor = getattr(cfg.logging, config_param)
        if ema_factor in train_state.ema_params:
            params = train_state.ema_params[ema_factor]
        else:
            print(
                f"Warning: EMA factor {ema_factor} not found in ema_params. Using instantaneous params."
            )
            params = train_state.params
    else:
        params = train_state.params

    return dist_utils.safe_unreplicate(cfg, params)


def _save_ckpt_on_signal(
    cfg: config_dict.ConfigDict, train_state: state_utils.EMATrainState
) -> None:
    save_state(train_state, cfg)
    sys.exit(0)


def register_signal_handlers(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
) -> None:
    """Drop a checkpoint on SIGTERM or SIGINT."""
    handler = functools.partial(_save_ckpt_on_signal, cfg, train_state)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def save_state(
    train_state: state_utils.EMATrainState,
    cfg: config_dict.ConfigDict,
) -> None:
    """Save flax training state."""
    with open(
        f"{cfg.logging.output_folder}/{cfg.logging.output_name}_{dist_utils.safe_index(cfg, train_state.step)//cfg.logging.save_freq}.pkl",
        "wb",
    ) as f:
        state = jax.device_get(dist_utils.safe_unreplicate(cfg, train_state))
        f.write(to_bytes(state))


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Compute the norm of the gradient pytree."""
    flat_params = ravel_pytree(grads)[0]
    return jnp.linalg.norm(flat_params)


def log_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    grads: jnp.ndarray,
    loss_value: float,
    loss_fn_args: Tuple,
    prng_key: jnp.ndarray,
    step_time: float,
) -> jnp.ndarray:
    """Log metrics to wandb, make a figure, and checkpoint the parameters."""

    grads = dist_utils.safe_unreplicate(cfg, grads)
    loss_value = dist_utils.safe_index(cfg, jnp.array(loss_value))
    step = dist_utils.safe_index(cfg, train_state.step)
    learning_rate = statics.schedule(step)

    metrics = {
        "loss": loss_value,
        "grad": compute_grad_norm(grads),
        "learning_rate": learning_rate,
        "step_time": step_time,
    }

    wandb.log(metrics)

    if (step % cfg.logging.visual_freq) == 0:
        prng_key = make_sample_plot(cfg, statics, train_state, prng_key)
        make_loss_fn_args_plot(cfg, statics, train_state, loss_fn_args)

    if (step % cfg.logging.save_freq) == 0:
        save_state(train_state, cfg)

    return prng_key


def make_sample_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> jnp.ndarray:
    """Plot generated samples against the target for low-dimensional problems."""
    batch_sample = flow_map.batch_sample
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")

    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    steps = [1, 2, 5, 10, 25]
    titles = ["base and target"] + [rf"${step}$-step" for step in steps]

    plot_x1s = next(statics.ds)[: cfg.logging.plot_bs]

    x0s = statics.sample_rho0(cfg.logging.plot_bs, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), cfg.logging.plot_bs, cfg.problem.d))
    for kk, step in enumerate(steps):
        xhats[kk] = batch_sample(
            train_state.apply_fn,
            params_for_visual,
            x0s,
            step,
            -jnp.ones(cfg.logging.plot_bs),
        )

    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax in axs.ravel():
        ax.set_xlim([-1.25, 1.25])
        ax.set_ylim([-1.25, 1.25])
        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    for jj in range(ncols):
        ax = axs[jj]
        ax.set_title(titles[jj], fontsize=fontsize)

        if jj == 0:
            ax.scatter(x0s[:, 0], x0s[:, 1], s=0.1, alpha=0.5, marker="o", c="black")
            ax.scatter(plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0")
        else:
            ax.scatter(plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0")
            ax.scatter(
                xhats[jj - 1, :, 0],
                xhats[jj - 1, :, 1],
                s=0.1,
                alpha=0.5,
                marker="o",
                c="black",
            )

    wandb.log({"samples": wandb.Image(fig)})
    return prng_key


def make_loss_fn_args_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
) -> None:
    """Plot the loss function arguments (time pairs and interpolated states)."""
    data_args = loss_fn_args[1:]
    (x0batch, x1batch, _, sbatch, tbatch, _, _, _) = (
        dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    )

    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    sbatch = jnp.squeeze(sbatch)
    tbatch = jnp.squeeze(tbatch)

    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    xtbatch = statics.interp.batch_calc_It(tbatch, x0batch, x1batch)

    if cfg.problem.d == 2:
        titles = [r"$x_0$", r"$x_1$", r"$x_t$", r"$(s, t)$"]
    else:
        titles = [r"$(s, t)$"]

    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )

    for kk, ax in enumerate(axs.ravel()):
        if kk == (len(titles) - 1):
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
        elif cfg.problem.d == 2:
            ax.set_xlim([-1.25, 1.25])
            ax.set_ylim([-1.25, 1.25])

        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    for jj in range(ncols):
        ax = axs[0, jj]
        ax.set_title(titles[jj], fontsize=fontsize)

        if cfg.problem.d == 2:
            if jj == 0:
                ax.scatter(x0batch[:, 0], x0batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 1:
                ax.scatter(x1batch[:, 0], x1batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 2:
                ax.scatter(xtbatch[:, 0], xtbatch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 3:
                ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")
        else:
            ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})
