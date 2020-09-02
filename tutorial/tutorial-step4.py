#!/usr/bin/python3

import os
import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from dftbplus_calc import DftbPlusCalculator


os.environ['DFTB_PREFIX'] = './'

# Coordinates taken from the following publication:
# Borlido et al., "The ground state of two-dimensional silicon",
#   2D Mater. 5, 035010 (2018), doi:10.1088/2053-1583/aab9ea
# and reoptimized with GPAW (LDA, 400 eV plane wave basis)
atoms = Atoms('Si10', 
              pbc=True,
              cell=(6.3245948276063171, 7.3351795905906787, 15., 90., 90., 90.),
              positions=[[2.5238234215085167,  1.8915376909714172,  7.5000000000042730],
                         [4.4402893598836890,  1.8913405945624528,  8.8460149135358996],
                         [1.2984521401199047,  3.8644391832367875,  7.5000000000030678],
                         [2.1934070307608962,  5.5591598065851038,  6.1539843200547191],
                         [4.4402893598687605,  1.8913405945582829,  6.1539850864607581],
                         [2.1934070307611422,  5.5591598065878962,  8.8460156799451646],
                         [5.3352644364352750,  0.1966184137746097,  7.4999999999998295],
                         [1.2984323446246357,  7.2538814320204086,  7.4999999999998987],
                         [5.3352446040859824,  3.5860618244812916,  7.4999999999982716],
                         [4.1098722380006052,  5.5589627502960530,  7.4999999999981020]])

calc = DftbPlusCalculator(atoms, kpts=(4, 3, 1), use_spline=True,
                          maximum_angular_momenta={'Si': 1})
atoms.set_calculator(calc)
atoms.set_calculator(calc)

# Local geometry optimization
ecf = ExpCellFilter(atoms, mask=(1, 1, 0, 0, 0, 1))
dyn = BFGS(ecf, logfile='-', trajectory='opt_zigzag_dumbbell.traj')
dyn.run(fmax=0.01)
write('relaxed_zigzag_dumbbell.traj', atoms)