import numpy as np
from ase.db import connect
from ase.build import bulk
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.calculator import kptdensity2monkhorstpack
from gpaw import GPAW, PW, Mixer, FermiDirac

calc = GPAW(mode=PW(400),
            xc='LDA',
            txt='gpawout_diamond.txt',
            occupations=FermiDirac(0.05),
            mixer=Mixer(0.1, 5, 100.),
            convergence={'density': 1e-5, 'eigenstates': 1e-5},
            )

db = connect('training.db')
Z = atomic_numbers['Si']
d0 = 2 * covalent_radii[Z]
a0 = d0 * 4. / np.sqrt(3)

for x in np.arange(0.9, 1.31, 0.05):
    atoms = bulk('Si', 'diamond', a=a0 * x)
    kpts = kptdensity2monkhorstpack(atoms, 2.5, even=False)
    calc.set(kpts=kpts)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    db.write(atoms, relaxed=1, gaid=0)
