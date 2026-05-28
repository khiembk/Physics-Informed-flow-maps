"""
Path encoding network phi(t, x0, x1) for the physics-informed interpolant:

    x_t = (1-t)*x0 + t*x1 + alpha(t) * phi(t, x0, x1),   alpha(t) = t*(1-t)

phi maps (t, x0, x1) -> correction field of same shape as x0.

Three options (config.phi_network_type):
  "unet"        -> PhiUNet          ~0.72M params  (C=3, 64x64)
  "transformer" -> PhiTransformer   ~0.90M params  (C=3, 64x64)
  "mlp"         -> PathEncodingMLP  low-dim only    (checker, d=2)

PhysicsInformedPath wraps phi and exposes x_t and dx_t/dt.
"""

from dataclasses import field
from typing import List, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from ml_collections import config_dict

from .edm2_net import (
    Block,
    MPConv,
    MPPositionalEmbedding,
    mp_cat,
    mp_silu,
    mp_sum,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _patchify(x: jnp.ndarray, P: int) -> jnp.ndarray:
    """(C, H, W) -> (N_patches, C*P*P)."""
    C, H, W = x.shape
    Nh, Nw = H // P, W // P
    x = x.reshape(C, Nh, P, Nw, P)
    x = jnp.transpose(x, (1, 3, 0, 2, 4))  # (Nh, Nw, C, P, P)
    return x.reshape(Nh * Nw, C * P * P)


def _unpatchify(x: jnp.ndarray, P: int, C_out: int, H: int, W: int) -> jnp.ndarray:
    """(N_patches, C_out*P*P) -> (C_out, H, W)."""
    Nh, Nw = H // P, W // P
    x = x.reshape(Nh, Nw, C_out, P, P)
    x = jnp.transpose(x, (2, 0, 3, 1, 4))  # (C_out, Nh, P, Nw, P)
    return x.reshape(C_out, H, W)


def _sinusoidal_emb(t: float, dim: int) -> jnp.ndarray:
    """Sinusoidal Fourier features for a scalar t, returns (dim,)."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
    t_f = jnp.asarray(t, dtype=jnp.float32)
    return jnp.concatenate([jnp.cos(t_f * freqs), jnp.sin(t_f * freqs)], axis=0)


# ---------------------------------------------------------------------------
# Option A: Tiny UNet (recommended, ~0.72M for C=3, 64x64)
# ---------------------------------------------------------------------------

class PhiUNet(nn.Module):
    """Tiny EDM2-style UNet: phi(t, x0, x1) -> correction (C_out, H, W).

    Input:  concatenated (x0, x1)  -> 2*C_in channels
    Output: correction field       -> C_out channels

    Recommended config (<=1M params for C<=4, 64x64):
        model_channels=16, channel_mult=(1,2,4), num_blocks=1, attn_resolutions=()
    """

    C_in: int                   # channels in a single state (e.g. 3 for NS)
    C_out: int                  # output channels (= C_in)
    img_resolution: int         # H = W (e.g. 64)
    model_channels: int         # base channel width (16)
    channel_mult: tuple         # e.g. (1, 2, 4)
    num_blocks: int             # residual blocks per resolution level (1)
    attn_resolutions: tuple     # resolutions with self-attention, () = none

    def setup(self):
        cblock = [self.model_channels * m for m in self.channel_mult]
        cst = cblock[0]
        cemb = max(cblock)
        n_levels = len(cblock)

        # single-time embedding (t only, not s)
        self.emb_fourier = MPPositionalEmbedding(cst)
        self.emb_linear = MPConv(cst, cemb)
        self.out_gain = self.param("out_gain", nn.initializers.zeros, ())

        # ---- encoder ----
        enc = {}
        skips = []          # channel sizes of each encoder output (for skip connections)
        cout = 2 * self.C_in + 1    # concatenated x0+x1 plus constant channel

        for level, ch in enumerate(cblock):
            res = self.img_resolution >> level
            if level == 0:
                enc[f"{res}x{res}_conv"] = MPConv(cout, ch, kernel=(3, 3))
                cout = ch
                skips.append(cout)
            else:
                enc[f"{res}x{res}_down"] = Block(
                    cout, cout, cemb, flavor="enc", resample_mode="down"
                )
                skips.append(cout)

            for i in range(self.num_blocks):
                cin = cout
                cout = ch
                enc[f"{res}x{res}_block{i}"] = Block(
                    cin, cout, cemb, flavor="enc",
                    attention=(res in self.attn_resolutions),
                )
                skips.append(cout)

        self.enc = enc

        # ---- decoder ----
        dec = {}

        for level in reversed(range(n_levels)):
            ch = cblock[level]
            res = self.img_resolution >> level

            if level == n_levels - 1:           # bottom
                dec[f"{res}x{res}_in0"] = Block(cout, cout, cemb, flavor="dec", attention=True)
                dec[f"{res}x{res}_in1"] = Block(cout, cout, cemb, flavor="dec")
            else:
                dec[f"{res}x{res}_up"] = Block(
                    cout, cout, cemb, flavor="dec", resample_mode="up"
                )

            for i in range(self.num_blocks + 1):
                skip = skips.pop()
                cin = cout + skip
                cout = ch
                dec[f"{res}x{res}_block{i}"] = Block(
                    cin, cout, cemb, flavor="dec",
                    attention=(res in self.attn_resolutions),
                )

        self.dec = dec
        self.out_conv = MPConv(cout, self.C_out, kernel=(3, 3))

    def __call__(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        """
        t  : scalar in [0,1]
        x0 : (C_in, H, W)
        x1 : (C_in, H, W)
        returns: (C_out, H, W)
        """
        t_arr = jnp.asarray(t, dtype=jnp.float32).reshape(1)
        x = jnp.concatenate([x0, x1], axis=0)[None]        # (1, 2*C, H, W)
        x = jnp.concatenate([x, jnp.ones_like(x[:, :1])], axis=1)  # + constant ch

        emb = mp_silu(self.emb_linear(self.emb_fourier(t_arr)))     # (1, cemb)

        # encoder
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb, train=train)
            skips.append(x)

        # decoder
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), dim=1)
            x = block(x, emb, train=train)

        x = self.out_conv(x, gain=self.out_gain)
        return x[0]     # (C_out, H, W)


# ---------------------------------------------------------------------------
# Option B: Lightweight Transformer (ViT-style, ~0.90M for C=3, 64x64)
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + FFN."""

    embed_dim: int
    num_heads: int
    ffn_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
        )(y, y)
        x = x + y
        # FFN
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.ffn_dim)(y)
        y = jax.nn.gelu(y)
        y = nn.Dense(self.embed_dim)(y)
        return x + y


class PhiTransformer(nn.Module):
    """Lightweight ViT: phi(t, x0, x1) -> correction (C_out, H, W).

    Patchifies (x0 cat x1), processes with transformer, unpatchifies.

    Recommended config (<=1M params for C<=4, 64x64):
        patch_size=8, embed_dim=128, num_heads=4, num_layers=6, ffn_dim=256
    """

    C_in: int
    C_out: int
    img_resolution: int     # H = W
    patch_size: int         # P, e.g. 8 -> 64 patches for 64x64
    embed_dim: int          # e.g. 128
    num_heads: int          # e.g. 4
    num_layers: int         # e.g. 6
    ffn_dim: int            # e.g. 256
    fourier_dim: int = 64   # dim of sinusoidal time embedding before MLP

    def setup(self):
        P = self.patch_size
        N = (self.img_resolution // P) ** 2

        self.patch_embed = nn.Dense(self.embed_dim)
        self.pos_emb = self.param(
            "pos_emb", nn.initializers.normal(0.02), (N, self.embed_dim)
        )
        self.time_fc1 = nn.Dense(self.embed_dim)
        self.time_fc2 = nn.Dense(self.embed_dim)
        self.blocks = [
            _TransformerBlock(self.embed_dim, self.num_heads, self.ffn_dim)
            for _ in range(self.num_layers)
        ]
        self.norm = nn.LayerNorm()
        self.out_proj = nn.Dense(self.C_out * P * P)

    def __call__(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        """
        t  : scalar in [0,1]
        x0 : (C_in, H, W)
        x1 : (C_in, H, W)
        returns: (C_out, H, W)
        """
        P = self.patch_size
        H = W = self.img_resolution

        # patchify concatenated input
        x = jnp.concatenate([x0, x1], axis=0)  # (2*C_in, H, W)
        x = _patchify(x, P)                     # (N, 2*C_in*P*P)

        # patch + positional embedding
        x = self.patch_embed(x) + self.pos_emb  # (N, embed_dim)

        # time embedding: sinusoidal -> 2-layer MLP
        t_emb = _sinusoidal_emb(t, self.fourier_dim)           # (fourier_dim,)
        t_emb = jax.nn.gelu(self.time_fc1(t_emb))              # (embed_dim,)
        t_emb = self.time_fc2(t_emb)                            # (embed_dim,)
        x = x + t_emb[None]                                     # broadcast over N

        # transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.out_proj(x)    # (N, C_out*P*P)

        return _unpatchify(x, P, self.C_out, H, W)     # (C_out, H, W)


# ---------------------------------------------------------------------------
# Option C: MLP for low-dimensional states (checker, d=2)
# ---------------------------------------------------------------------------

class PathEncodingMLP(nn.Module):
    """MLP phi(t, x0, x1) for flat low-dim states.

    Input concatenates [t, x0, x1] -> 2*d+1 dims.

    Recommended config (<=1M params, d=2):
        n_hidden=4, n_neurons=384
    """

    d: int          # dimension of x0 (= x1)
    n_hidden: int   # number of hidden layers
    n_neurons: int  # neurons per hidden layer

    @nn.compact
    def __call__(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        inp = jnp.concatenate([jnp.asarray(t, dtype=jnp.float32).reshape(1), x0, x1])
        x = jax.nn.gelu(nn.Dense(self.n_neurons)(inp))
        for _ in range(self.n_hidden):
            x = jax.nn.gelu(nn.Dense(self.n_neurons)(x))
        return nn.Dense(self.d)(x)


# ---------------------------------------------------------------------------
# Physics-informed path wrapper
# ---------------------------------------------------------------------------

class PhysicsInformedPath(nn.Module):
    """Wraps phi to compute the physics-informed interpolant and its velocity.

    Path:     x_t = (1-t)*x0 + t*x1 + alpha(t)*phi(t,x0,x1)
    Velocity: dx_t/dt = (x1-x0) + alpha_dot(t)*phi + alpha(t)*d_phi/dt

    where alpha(t) = t*(1-t), alpha_dot(t) = 1-2*t.
    """

    phi: nn.Module  # any of PhiUNet / PhiTransformer / PathEncodingMLP

    def __call__(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        """Returns x_t."""
        alpha = t * (1.0 - t)
        correction = self.phi(t, x0, x1, train=train)
        return (1.0 - t) * x0 + t * x1 + alpha * correction

    def velocity(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        train: bool = False,
        eps_t: float = 1e-3,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (x_t, dx_t/dt).

        d_phi/dt is approximated with central finite differences to avoid
        jax.jvp through cuDNN conv kernels (CUDNN_STATUS_INTERNAL_ERROR on H100).
        Two forward passes; error is O(eps_t^2) ≈ 1e-6.
        """
        alpha     = t * (1.0 - t)
        alpha_dot = 1.0 - 2.0 * t

        # Central finite difference for dphi/dt
        t_lo = jnp.clip(t - eps_t, 0.0, 1.0)
        t_hi = jnp.clip(t + eps_t, 0.0, 1.0)
        phi_hi = self.phi(t_hi, x0, x1, train=train)
        phi_lo = self.phi(t_lo, x0, x1, train=train)
        phi_val = (phi_hi + phi_lo) / 2.0
        dphi_dt = (phi_hi - phi_lo) / (t_hi - t_lo + 1e-12)

        xt = (1.0 - t) * x0 + t * x1 + alpha * phi_val
        vt = (x1 - x0) + alpha_dot * phi_val + alpha * dphi_dt
        return xt, vt


# ---------------------------------------------------------------------------
# Factory + initializer
# ---------------------------------------------------------------------------

def setup_path_encoding(cfg: config_dict.ConfigDict) -> nn.Module:
    """Build the PhysicsInformedPath wrapper from config.

    cfg fields (all under cfg.phi):
      network_type : "unet" | "transformer" | "mlp"

      For "unet":
        C_in, C_out, img_resolution, model_channels,
        channel_mult, num_blocks, attn_resolutions

      For "transformer":
        C_in, C_out, img_resolution, patch_size,
        embed_dim, num_heads, num_layers, ffn_dim

      For "mlp":
        d, n_hidden, n_neurons
    """
    phi_cfg = cfg.phi
    ntype = phi_cfg.network_type

    if ntype == "unet":
        phi = PhiUNet(
            C_in=phi_cfg.C_in,
            C_out=phi_cfg.C_out,
            img_resolution=phi_cfg.img_resolution,
            model_channels=phi_cfg.model_channels,
            channel_mult=tuple(phi_cfg.channel_mult),
            num_blocks=phi_cfg.num_blocks,
            attn_resolutions=tuple(phi_cfg.attn_resolutions),
        )
    elif ntype == "transformer":
        phi = PhiTransformer(
            C_in=phi_cfg.C_in,
            C_out=phi_cfg.C_out,
            img_resolution=phi_cfg.img_resolution,
            patch_size=phi_cfg.patch_size,
            embed_dim=phi_cfg.embed_dim,
            num_heads=phi_cfg.num_heads,
            num_layers=phi_cfg.num_layers,
            ffn_dim=phi_cfg.ffn_dim,
        )
    elif ntype == "mlp":
        phi = PathEncodingMLP(
            d=phi_cfg.d,
            n_hidden=phi_cfg.n_hidden,
            n_neurons=phi_cfg.n_neurons,
        )
    else:
        raise ValueError(f"Unknown phi network_type: {ntype}")

    return PhysicsInformedPath(phi=phi)


def initialize_path_encoding(
    cfg: config_dict.ConfigDict,
    ex_x: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[nn.Module, dict, jnp.ndarray]:
    """Initialize path encoding network and return (model, params, prng_key).

    ex_x : example state array (d,) or (C, H, W)
    """
    model = setup_path_encoding(cfg)

    ex_t = 0.5
    prng_key, init_key = jax.random.split(prng_key)
    params = model.init(init_key, ex_t, ex_x, ex_x)

    n_params = ravel_pytree(params)[0].size
    print(f"Path encoding phi params: {n_params:,}  ({n_params/1e6:.3f}M)")
    return model, params, prng_key
