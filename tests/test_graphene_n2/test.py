''' Tests the use of 'referencing', using N2
on a graphene sheet as an example.

By specifying the separate subsystems as 'references',
only the C-N interaction will be fitted, so as to
reproduce the interaction energy of the N2 with
the substrate.

The reference data is calculated using DFTB
with a simple polynomial form of the repulsive
potential. The fitted DFTB model, which uses the
same electronic parameters, should therefore
exactly reproduce the training data.

This test requires *-*_no_repulsion.skf and
*-*_spline.skf files as created by the
generate_skf.py script.
'''
import os
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.build import molecule
from ase.lattice.hexagonal import Graphene
from ase.db import connect
from tango import TANGO
from tango.relax_utils import finalize
from tango.calculators.dftbplus_calc import DftbPlusCalculator
from tango.utilities import convert_array


elements = ['C', 'N']
for el_a in elements:
    for el_b in elements:
        f = '%s-%s_no_repulsion.skf' % (el_a, el_b)
        assert os.path.exists(f), 'Run generate_skf.py script!'


class DftbPlusCalc(DftbPlusCalculator):
    def __init__(self, *args, **kwargs):
        kwargs['Hamiltonian_SCC'] = 'No'
        kwargs['Hamiltonian_ShellResolvedSCC'] = 'No'
        kwargs['Hamiltonian_OrbitalResolvedSCC'] = 'No'
        DftbPlusCalculator.__init__(self, *args, **kwargs)


def calculate_target_repulsion(atoms, rc):
    N = len(atoms)
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()

    e, f = 0., np.zeros((N, 3))
    coeff = np.array([0., 0., 0., 14., -5., 2.])
    powers = np.arange(len(coeff))

    for i in range(N):
        for j in range(N):
            r = atoms.get_distance(i, j)
            if sym[i] == sym[j] or r > rc:
                continue
            e += 0.5 * (coeff * (rc - r) ** powers).sum()
            dedr = -coeff[1:] * powers[1:] * (rc - r) ** (powers[1:] - 1)
            drdx = (pos[i] - pos[j]) / r
            f[i] += -dedr.sum() * drdx
    return e, f


def generate_database(dbfile, rc):
    np.random.seed(123)
    db = connect(dbfile)

    rcov = covalent_radii[atomic_numbers['C']]
    lc = 2 * np.sqrt(3) * rcov
    slab = Graphene(symbol='C', latticeconstant=[lc, 12.]).repeat((3, 3, 1))
    slab.center()

    kwargs = {'maximum_angular_momenta': mam, 'use_spline': False,
              'Hamiltonian_SlaterKosterFiles_Suffix': '"_no_repulsion.skf"'}
    calc = DftbPlusCalc(slab, **kwargs)
    slab.set_calculator(calc)
    e_ref_slab = slab.get_potential_energy()
    f_ref_slab = slab.get_forces()

    m = molecule('N2')
    m.positions += np.array([2.5, 3., 7.5])

    for i in range(10):
        print('Generating structure %d' % i)
        n2 = m.copy()
        for i, axis in enumerate('xyz'):
            angle = np.random.random() * 180.
            n2.rotate(angle, axis, center='COP')

        calc = DftbPlusCalc(n2, **kwargs)
        n2.set_calculator(calc)
        e_ref_n2 = n2.get_potential_energy()
        f_ref_n2 = n2.get_forces()

        atoms = slab + n2
        calc = DftbPlusCalc(atoms, **kwargs)
        atoms.set_calculator(calc)
        e = atoms.get_potential_energy()
        f = atoms.get_forces()

        e_rep, f_rep = calculate_target_repulsion(atoms, rc)
        finalize(atoms, energy=e+e_rep, forces=f+f_rep, stress=None)

        e_dft_ref = convert_array([e_ref_slab, e_ref_n2])
        f_dft_ref = convert_array(np.vstack((f_ref_slab, f_ref_n2)))
        db.write(atoms, relaxed=1, e_dft_ref=e_dft_ref, f_dft_ref=f_dft_ref)

    return


rc, rmin = 2.25, 0.5
pair = 'C-N'
rcuts = {pair: rc, 'N-N': None, 'C-C': None}
mam = {'C': 1, 'N': 1}
dbfile = 'training.json'

if os.path.exists(dbfile):
    os.remove(dbfile)
generate_database(dbfile, rc)

for mode in ['poly', 'exp_poly', 'exp_spline']:
    powers = {pair: range(2, 7) if 'poly' in mode else range(4)}
    rmins = {pair: rmin} if 'exp' in mode else None
    calc = TANGO(elements,
                 DftCalc=None,
                 DftbPlusCalc=DftbPlusCalc,
                 mode=mode,
                 kptdensity=None,
                 maximum_angular_momenta=mam,
                 fit_constant=None,
                 kBT=2.,
                 force_scaling=1.,
                 rcuts=rcuts,
                 rmins=rmins,
                 powers=powers,
                 referencing=[range(18), [18, 19]],
                 )

    residual = calc.fit_repulsion([dbfile], run_checks=False)
    eps = 1e-6 if 'poly' in mode else 1e-2
    assert residual < eps, (mode, residual, eps)
    os.system('mv %s.pdf %s_%s.pdf' % (pair, pair, mode))
