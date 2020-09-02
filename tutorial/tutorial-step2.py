#!/usr/bin/python3

from ase.data import atomic_numbers, covalent_radii
from dftbplus_calc import DftbPlusCalculator
from tango import TANGO

elements = ['Si']
mam = {'Si': 1}           # the maximum angular momenta for each element
mode = 'exp_poly'         # the functional form (exponential decay at short distances, then a polynomial)
fit_constant = 'element'  # allow for one constant energy shift for each element
kBT = 2.0                 # energy which defines the Boltzmann weights from the (relative) cohesive energies
kptdensity = 2.5          # the k-point density in points per Ang^-1
dbfiles = ['training.db']
force_scaling = 0.0       # how the forces should be weighted w.r.t. the forces;
                          # this does not matter here because the forces are zero by symmetry

rcov = covalent_radii[atomic_numbers['Si']]
rcuts = {'Si-Si': 1.5 * 2 * rcov}  # 'outer' cutoff as approximately 1.5 times the NN distance
rmins = {'Si-Si': 0.8 * 2 * rcov}  # 'inner' cutoff below which we switch to exponential repulsion
powers = {'Si-Si': range(2, 7)}    # standard (rcut-r)**2 + ... + (rcut-r)**6 polynomial

calc = TANGO(elements,
             DftbPlusCalc=DftbPlusCalculator,
             kptdensity=kptdensity,
             rmins=rmins,
             rcuts=rcuts,
             powers=powers,
             fit_constant=fit_constant,
             kBT=kBT,
             mode=mode,
             force_scaling=force_scaling,
             maximum_angular_momenta=mam,
             )

residual = calc.fit_repulsion(dbfiles, run_checks=True)
print('Residual:', residual)