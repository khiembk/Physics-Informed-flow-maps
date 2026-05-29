from configs.pde_configs import get_phase2_config
def get_config(dataset_location="", output_folder="", phi_ckpt=""):
    return get_phase2_config("navier_stokes_2d", dataset_location, output_folder, phi_ckpt)
