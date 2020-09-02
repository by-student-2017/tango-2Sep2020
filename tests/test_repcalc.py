import os
import numpy as np
from ase import Atoms
from tango.calculators.repulsive_potential import RepulsivePotential
from tango.relax_utils import relax_standard

lines = '''Spline
11 3.87771801
3.90696773 3.06926162 0.14584880
1.29257267 1.38705898 0.28381049 -0.54051635 0.24643768 0.00333443
1.38705898 1.65330039 0.23494202 -0.49385707 0.24738285 -0.00489104
1.65330039 1.91954180 0.12090012 -0.36317005 0.24347626 -0.02148938
1.91954180 2.18578322 0.04106235 -0.23809291 0.22631217 -0.05054463
2.18578322 2.45202463 -0.00723972 -0.12833406 0.18594095 -0.09318512
2.45202463 2.71826605 -0.02998586 -0.04913983 0.11151174 -0.10903995
2.71826605 2.98450746 -0.03722231 -0.01294947 0.02441889 0.05574416
2.98450746 3.25074888 -0.03788705 0.01190736 0.06894310 0.04527726
3.25074888 3.51699029 -0.02897533 0.05824674 0.10510715 -0.22602168
3.51699029 3.78323170 -0.01028274 0.06615020 -0.07542185 -0.11063619
3.78323170 3.87771801 -0.00010502 0.00246219 -0.16378965 1.81772166 11.5 -147.8
'''
tmpfile = '.tmp.skf'
with open(tmpfile, 'w') as f:
    f.write(lines)

cell = np.identity(3) * [2, 3, 4]
np.random.seed(666)
positions = np.dot(np.random.random((8, 3)), cell.T)
atoms = Atoms('C8', cell=cell, positions=positions, pbc=True)

calc = RepulsivePotential(atoms, skfdict={'C-C': tmpfile})
atoms = relax_standard(atoms, calc, fmax=5e-2, smax=1e-4, variable_cell=False,
                       trajfile=None, logfile='-', maxsteps=10000,
                       verbose=False, dEmin=1e-3, optimizer='BFGSLineSearch',
                       ucf=False)

e = atoms.get_potential_energy()
assert abs(e - -38.221711) < 5e-3
os.remove(tmpfile)
