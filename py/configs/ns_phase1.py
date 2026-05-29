from configs.pde_configs import get_phase1_config
def get_config(dataset_location="", output_folder=""):
    return get_phase1_config("navier_stokes_2d", dataset_location, output_folder)
