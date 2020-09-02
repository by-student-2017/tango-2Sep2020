#!/usr/bin/python3

import os
import numpy as np
from ase.io import write
from ase.lattice.hexagonal import Graphene
from ase.optimize import BFGS
from ase.constraints import ExpCellFilter
from ase.data import atomic_numbers, covalent_radii
from dftbplus_calc import DftbPlusCalculator

os.environ['DFTB_PREFIX'] = './'

rcov = covalent_radii[atomic_numbers['Si']]
lc = 2 * np.sqrt(3) * rcov
atoms = Graphene(symbol='Si', latticeconstant=[lc, 12.])
atoms.center()

calc = DftbPlusCalculator(atoms, kpts=(5, 5, 1), use_spline=True,
                          maximum_angular_momenta={'Si': 1})
atoms.set_calculator(calc)

ecf = ExpCellFilter(atoms, mask=(1, 1, 0, 0, 0, 1))
dyn = BFGS(ecf, logfile='-')
dyn.run(fmax=0.01)
write('relaxed_silicene_flat.traj', atoms)

atoms.positions[:,2] += np.array([0.5, -0.5])
atoms.set_calculator(calc)
dyn = BFGS(ecf, logfile='-')
dyn.run(fmax=0.01)
write('relaxed_silicene_buckled.traj', atoms)