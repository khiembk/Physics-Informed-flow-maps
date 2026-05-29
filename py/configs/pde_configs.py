"""
Shared config factory for all 4 PDE systems.

Usage in config wrappers:
    from configs.pde_configs import get_phase1_config, get_phase2_config, get_baseline_config
"""

import os
import ml_collections

# ---------------------------------------------------------------------------
# Per-system parameters
# ---------------------------------------------------------------------------

SYSTEM_PARAMS = {
    "shallow_water_2d": {
        "C": 3,
        "channels": ["eta", "m_x", "m_y"],
        "pde": {"g": 1.0, "nu": 0.002},
    },
    "navier_stokes_2d": {
        "C": 3,
        "channels": ["c", "u_x", "u_y"],
        "pde": {"nu": 0.001, "kappa": 0.0005},
    },
    "mhd_2d": {
        "C": 4,
        "channels": ["u_x", "u_y", "B_x", "B_y"],
        "pde": {"nu": 0.001},
    },
    "multiphase_2d": {
        "C": 3,
        "channels": ["P", "S_w", "S_o"],
        "pde": {},
    },
}


def _network_unet(C, model_channels=64, attn_res=16):
    cfg = ml_collections.ConfigDict()
    cfg.network_type    = "edm2"
    cfg.img_resolution  = 64
    cfg.img_channels    = C
    cfg.label_dim       = 0
    cfg.logvar_channels = 128
    cfg.rescale         = 0.5
    cfg.use_weight      = False
    cfg.use_bfloat16    = False
    cfg.use_cfg         = False
    cfg.load_path       = ""
    cfg.load_ema_fac    = None
    cfg.reset_optimizer = True
    cfg.input_dims      = (C, 64, 64)
    cfg.output_dim      = None
    cfg.n_hidden        = None
    cfg.n_neurons       = None
    cfg.act             = None
    cfg.use_residual    = None
    cfg.num_classes     = None
    cfg.unet_kwargs     = {
        "model_channels":    model_channels,
        "channel_mult":      [1, 2, 4],
        "num_blocks":        1,
        "attn_resolutions":  [attn_res] if attn_res else [],
        "channel_mult_noise": None,
        "channel_mult_emb":  None,
        "block_kwargs":      {"dropout": 0.0},
    }
    return cfg


def _phi_config(C):
    cfg = ml_collections.ConfigDict()
    cfg.network_type    = "unet"
    cfg.C_in            = C
    cfg.C_out           = C
    cfg.img_resolution  = 64
    cfg.model_channels  = 16
    cfg.channel_mult    = (1, 2, 4)
    cfg.num_blocks      = 1
    cfg.attn_resolutions = ()
    cfg.ckpt            = ""
    return cfg


# ---------------------------------------------------------------------------
# Phase 1 config
# ---------------------------------------------------------------------------

def get_phase1_config(system: str, dataset_location: str = "",
                      output_folder: str = "") -> ml_collections.ConfigDict:
    p = SYSTEM_PARAMS[system]
    C = p["C"]
    config = ml_collections.ConfigDict()

    config.problem = ml_collections.ConfigDict()
    config.problem.system           = system
    config.problem.dataset_location = dataset_location
    config.problem.H, config.problem.W = 64, 64
    config.problem.C                = C
    config.problem.channel_names    = p["channels"]
    for k, v in p["pde"].items():
        setattr(config.problem, k, v)
    config.problem.tmin, config.problem.tmax = 0.0, 1.0

    config.phi = _phi_config(C)

    config.phase1 = ml_collections.ConfigDict()
    config.phase1.w0        = 1.0
    config.phase1.w_alpha   = 1.0
    config.phase1.lambda_sm = 0.01
    config.phase1.t_min     = 0.0
    config.phase1.t_max     = 1.0

    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs            = 32
    config.optimization.learning_rate = 1e-3
    config.optimization.total_steps   = 1_500
    config.optimization.warmup_steps  = 100
    config.optimization.clip          = 1.0
    config.optimization.seed          = 0

    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project = "physics-informed-flow-maps"
    config.logging.wandb_name    = f"{system}_phase1"
    config.logging.wandb_entity  = os.getenv("WANDB_ENTITY", "khiembk")
    config.logging.output_folder = output_folder
    config.logging.log_freq      = 100
    config.logging.save_freq     = 500

    return config


# ---------------------------------------------------------------------------
# Phase 2 config
# ---------------------------------------------------------------------------

def get_phase2_config(system: str, dataset_location: str = "",
                      output_folder: str = "",
                      phi_ckpt: str = "") -> ml_collections.ConfigDict:
    p = SYSTEM_PARAMS[system]
    C = p["C"]
    config = ml_collections.ConfigDict()

    config.problem = ml_collections.ConfigDict()
    config.problem.system           = system
    config.problem.dataset_location = dataset_location
    config.problem.H, config.problem.W = 64, 64
    config.problem.C                = C
    config.problem.channel_names    = p["channels"]
    config.problem.tmin, config.problem.tmax = 0.0, 1.0

    config.phi    = _phi_config(C)
    config.phi.ckpt = phi_ckpt

    config.network = _network_unet(C, model_channels=64, attn_res=16)

    config.training = ml_collections.ConfigDict()
    config.training.seed = 0

    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs            = 32
    config.optimization.learning_rate = 1e-4
    config.optimization.total_steps   = 10_000
    config.optimization.warmup_steps  = 500
    config.optimization.clip          = 1.0

    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project = "physics-informed-flow-maps"
    config.logging.wandb_name    = f"{system}_phase2"
    config.logging.wandb_entity  = os.getenv("WANDB_ENTITY", "khiembk")
    config.logging.output_folder = output_folder
    config.logging.log_freq      = 200
    config.logging.save_freq     = 2_000

    return config


# ---------------------------------------------------------------------------
# Baseline config
# ---------------------------------------------------------------------------

def get_baseline_config(system: str, dataset_location: str = "",
                         output_folder: str = "") -> ml_collections.ConfigDict:
    p = SYSTEM_PARAMS[system]
    C = p["C"]
    config = ml_collections.ConfigDict()

    config.problem = ml_collections.ConfigDict()
    config.problem.system           = system
    config.problem.dataset_location = dataset_location
    config.problem.H, config.problem.W = 64, 64
    config.problem.C                = C
    config.problem.channel_names    = p["channels"]
    config.problem.interp_type      = "linear"
    config.problem.tmin, config.problem.tmax = 0.0, 1.0

    config.network = _network_unet(C, model_channels=64, attn_res=16)

    config.training = ml_collections.ConfigDict()
    config.training.loss_type    = "diagonal"
    config.training.ema_facs     = [0.999, 0.9999]
    config.training.seed         = 0
    config.training.conditional  = False
    config.training.class_dropout = 0.0

    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs            = 32
    config.optimization.learning_rate = 1e-4
    config.optimization.total_steps   = 10_000
    config.optimization.warmup_steps  = 500
    config.optimization.clip          = 1.0

    config.logging = ml_collections.ConfigDict()
    config.logging.wandb_project = "physics-informed-flow-maps"
    config.logging.wandb_name    = f"{system}_baseline"
    config.logging.wandb_entity  = os.getenv("WANDB_ENTITY", "khiembk")
    config.logging.output_folder = output_folder
    config.logging.log_freq      = 200
    config.logging.save_freq     = 2_000
    config.logging.visual_freq   = 5_000

    return config
