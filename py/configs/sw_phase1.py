"""
Phase 1 config: shallow water equations — physics-informed path training.

Trains the phi (path encoding) network to minimize:
    L_path = L_phy + lambda_sm * L_sm

Uses shallow_water_2d dataset (easiest: only 1% relative change x0->xT).
"""

import os
import ml_collections


def get_config(dataset_location: str = "", output_folder: str = "") -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # ----- problem -----
    config.problem = ml_collections.ConfigDict()
    config.problem.system        = "shallow_water_2d"
    config.problem.dataset_location = dataset_location
    config.problem.H             = 64
    config.problem.W             = 64
    config.problem.C             = 3          # [eta, m_x, m_y]
    config.problem.channel_names = ["eta", "m_x", "m_y"]
    # PDE parameters (must match generator config)
    config.problem.g             = 1.0
    config.problem.nu            = 0.002

    # ----- phi network -----
    config.phi = ml_collections.ConfigDict()
    config.phi.network_type      = "unet"
    config.phi.C_in              = 3
    config.phi.C_out             = 3
    config.phi.img_resolution    = 64
    config.phi.model_channels    = 16
    config.phi.channel_mult      = (1, 2, 4)
    config.phi.num_blocks        = 1
    config.phi.attn_resolutions  = ()

    # ----- Phase 1 loss -----
    config.phase1 = ml_collections.ConfigDict()
    config.phase1.w0             = 1.0    # base weight w(t) = w0 + w_alpha * t
    config.phase1.w_alpha        = 1.0    # weight ramps to 2.0 at t=1
    config.phase1.lambda_sm      = 0.01   # spatial roughness coefficient
    config.phase1.t_min          = 0.0
    config.phase1.t_max          = 1.0

    # ----- optimization -----
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs           = 32        # pairs per step (x0,xT)
    config.optimization.learning_rate = 1e-3
    config.optimization.total_steps  = 50_000
    config.optimization.warmup_steps = 1_000
    config.optimization.clip         = 1.0
    config.optimization.seed         = 42

    # ----- logging -----
    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project  = "physics-informed-flow-maps"
    config.logging.wandb_name     = "sw_phase1_unet16"
    config.logging.wandb_entity   = os.getenv("WANDB_ENTITY", "khiembk")
    config.logging.output_folder  = output_folder
    config.logging.log_freq       = 500
    config.logging.save_freq      = 5_000

    return config
