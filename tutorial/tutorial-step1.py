#!/usr/bin/python3

import os
import numpy as np
from ase.db import connect
from ase.build import bulk
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.singlepoint import SinglePointCalculator

dbfile = 'training.db'
assert not os.path.exists(dbfile), 'Please remove the existing training.db file!'
db = connect(dbfile)

rcov = covalent_radii[atomic_numbers['Si']]
a0 = 2 * rcov * 4. / np.sqrt(3)

# Different fractions we will multiply the 'a0' lattice constant with:
fractions = np.arange(0.9, 1.31, 0.05)
# Associated total energies calculated with GPAW (LDA, 400 eV plane wave basis):
energies = [-8.156546300399613, -10.424474986302807, -11.546688457623372,
            -11.884715210546915, -11.648474045051344, -11.166297858643642,
            -10.518097847630376, -9.800655996150622, -9.06797134883969]
f = np.zeros((2, 3))

for x, e in zip(fractions, energies):
    atoms = bulk('Si', 'diamond', a=a0 * x)
    calc = SinglePointCalculator(atoms, energy=e, forces=f)
    atoms.set_calculator(calc)    
    db.write(atoms, relaxed=1, gaid=0)

print('Done!')