"""
Baseline config: shallow water — standard flow matching with linear interpolant.

Trains FlowMap v_theta (EDM2 UNet, ~12.28M params) to minimize the
diagonal flow-matching loss using a plain linear interpolant:

    x_t = (1-t)*x0 + t*xT
    target = xT - x0

No physics constraint, no path correction (phi=0). Serves as comparison
against the physics-informed Phase 1 + Phase 2 approach.
"""

import os
import ml_collections


def get_config(dataset_location: str = "", output_folder: str = "") -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # ----- problem -----
    config.problem = ml_collections.ConfigDict()
    config.problem.system           = "shallow_water_2d"
    config.problem.dataset_location = dataset_location
    config.problem.H                = 64
    config.problem.W                = 64
    config.problem.C                = 3
    config.problem.channel_names    = ["eta", "m_x", "m_y"]
    config.problem.interp_type      = "linear"
    config.problem.tmin             = 0.0
    config.problem.tmax             = 1.0

    # ----- flow map network (v_theta, EDM2 UNet ~12.28M params) -----
    config.network = ml_collections.ConfigDict()
    config.network.network_type     = "edm2"
    config.network.img_resolution   = 64
    config.network.img_channels     = 3
    config.network.label_dim        = 0
    config.network.logvar_channels  = 128
    config.network.rescale          = 0.5
    config.network.use_weight       = False
    config.network.use_bfloat16     = False
    config.network.use_cfg          = False
    config.network.load_path        = ""
    config.network.load_ema_fac     = None
    config.network.reset_optimizer  = True
    config.network.input_dims       = (3, 64, 64)
    config.network.output_dim       = None
    config.network.n_hidden         = None
    config.network.n_neurons        = None
    config.network.act              = None
    config.network.use_residual     = None
    config.network.num_classes      = None
    config.network.unet_kwargs = {
        "model_channels":    64,
        "channel_mult":      [1, 2, 4],
        "num_blocks":        1,
        "attn_resolutions":  [16],
        "channel_mult_noise": None,
        "channel_mult_emb":  None,
        "block_kwargs":      {"dropout": 0.0},
    }

    # ----- training -----
    config.training = ml_collections.ConfigDict()
    config.training.loss_type      = "diagonal"   # pure flow matching
    config.training.ema_facs       = [0.999, 0.9999]
    config.training.seed           = 0
    config.training.conditional    = False
    config.training.class_dropout  = 0.0

    # ----- optimization -----
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs            = 32
    config.optimization.learning_rate = 1e-4
    config.optimization.total_steps   = 10_000   # match Phase 2 for fair comparison
    config.optimization.warmup_steps  = 500
    config.optimization.clip          = 1.0
    config.optimization.decay_steps   = 10_000
    config.optimization.schedule_type = "cosine"

    # ----- logging -----
    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project  = "physics-informed-flow-maps"
    config.logging.wandb_name     = "sw_baseline_linear"
    config.logging.wandb_entity   = os.getenv("WANDB_ENTITY", "khiembk")
    config.logging.output_folder  = output_folder
    config.logging.log_freq       = 200
    config.logging.save_freq      = 10_000
    config.logging.visual_freq    = 5_000

    return config
