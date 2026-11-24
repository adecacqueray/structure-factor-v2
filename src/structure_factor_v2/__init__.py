"""
structure_factor_v2: structure factor S(k) of point sets in the unit cube [0, 1)^D, and plots.

>>> import structure_factor_v2 as sf
>>> points = sf.jittered_lattice(10_000, 2)          # a hyperuniform test pattern
>>> kint, S = sf.structure_factor(points)            # raw scattering intensity
>>> sf.plot_structure_factor(points)                 # log-log plot, radial average
"""
from .lattices import curled_lattice, jittered_lattice, korobov_lattice
from .structure_factor import (
    structure_factor,
    structure_factor_and_average,
)
from .viz import plot, plot_structure_factor

__all__ = [
    "structure_factor", "structure_factor_and_average", 
    "plot", "plot_structure_factor", "jittered_lattice", "curled_lattice", "korobov_lattice",
]

__version__ = "2.1.0"
