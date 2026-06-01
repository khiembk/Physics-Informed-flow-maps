from .ns_boussinesq import generate as generate_ns
from .mhd_2d import generate as generate_mhd
from .multiphase_2d import generate as generate_multiphase
from .shallow_water_2d import generate as generate_shallow_water
from .euler_2d import generate as generate_euler

SOLVERS = {
    "navier_stokes_2d": generate_ns,
    "mhd_2d":           generate_mhd,
    "multiphase_2d":    generate_multiphase,
    "shallow_water_2d": generate_shallow_water,
    "euler_2d":         generate_euler,
}
