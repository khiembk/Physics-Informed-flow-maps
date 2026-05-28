"""
Phase 2 config: train flow map v_theta on the physics-informed path from frozen phi.

Loads the best Phase 1 checkpoint (step 35k, lowest L_phy before L_sm blowup)
and trains v_theta (EDM2 UNet, 12.28M) with 10K steps of diagonal flow matching.
"""

import os
import ml_collections


def get_config(dataset_location: str = "", output_folder: str = "",
               phi_ckpt: str = "") -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # ----- problem -----
    config.problem = ml_collections.ConfigDict()
    config.problem.system        = "shallow_water_2d"
    config.problem.dataset_location = dataset_location
    config.problem.H             = 64
    config.problem.W             = 64
    config.problem.C             = 3
    config.problem.channel_names = ["eta", "m_x", "m_y"]
    config.problem.tmin          = 0.0
    config.problem.tmax          = 1.0

    # ----- frozen phi (Phase 1 checkpoint) -----
    config.phi = ml_collections.ConfigDict()
    config.phi.network_type     = "unet"
    config.phi.C_in             = 3
    config.phi.C_out            = 3
    config.phi.img_resolution   = 64
    config.phi.model_channels   = 16
    config.phi.channel_mult     = (1, 2, 4)
    config.phi.num_blocks       = 1
    config.phi.attn_resolutions = ()
    config.phi.ckpt             = phi_ckpt   # path to phi_step35000.npz

    # ----- flow map v_theta (EDM2 UNet ~12.28M) -----
    config.network = ml_collections.ConfigDict()
    config.network.network_type    = "edm2"
    config.network.img_resolution  = 64
    config.network.img_channels    = 3
    config.network.label_dim       = 0
    config.network.logvar_channels = 128
    config.network.rescale         = 0.5
    config.network.use_weight      = False
    config.network.use_bfloat16    = False
    config.network.use_cfg         = False
    config.network.load_path       = ""
    config.network.load_ema_fac    = None
    config.network.reset_optimizer = True
    config.network.input_dims      = (3, 64, 64)
    config.network.output_dim      = None
    config.network.n_hidden        = None
    config.network.n_neurons       = None
    config.network.act             = None
    config.network.use_residual    = None
    config.network.num_classes     = None
    config.network.unet_kwargs     = {
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
    config.training.seed = 0

    # ----- optimization -----
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs            = 32
    config.optimization.total_steps   = 10_000
    config.optimization.learning_rate = 1e-4
    config.optimization.warmup_steps  = 500
    config.optimization.clip          = 1.0

    # ----- logging -----
    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project = "physics-informed-flow-maps"
    config.logging.wandb_name    = "sw_phase2_pi_path"
    config.logging.wandb_entity  = os.getenv("WANDB_ENTITY", "khiembk")
    config.logging.output_folder = output_folder
    config.logging.log_freq      = 200
    config.logging.save_freq     = 2_000

    return config
