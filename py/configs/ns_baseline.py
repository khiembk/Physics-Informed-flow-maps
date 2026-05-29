from configs.pde_configs import get_baseline_config
def get_config(dataset_location="", output_folder=""):
    return get_baseline_config("navier_stokes_2d", dataset_location, output_folder)
