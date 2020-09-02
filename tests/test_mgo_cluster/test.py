''' Repulsion fitting test on (MgO)4 clusters.

The reference data is calculated using DFTB
with a simple polynomial form of the repulsive
potential. The fitted DFTB model, which uses the
same electronic parameters, should therefore
exactly reproduce the training data.

This test requires *-*_no_repulsion.skf files
as created by the generate_skf.py script.
'''
import os
import numpy as np
from ase import Atoms
from ase.db import connect
from tango import TANGO
from tango.relax_utils import finalize
from tango.calculators.dftbplus_calc import DftbPlusCalculator


elements = ['Mg', 'O']
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


def generate_database(dbfile, d, rc):
    np.random.seed(123)
    db = connect(dbfile)
    L = 12.
    x0 = (L - d) / 2.
    sym = ['Mg', 'O', 'O', 'Mg', 'O', 'Mg', 'Mg', 'O']
    cell = np.ones(3) * L
    pos = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pos.append([x0 + i * d, x0 + j * d, x0 + k * d])

    assert rc > d, (rc, d)
    dispmax = 0.5 * (rc - d) / np.sqrt(3)

    for i in range(10):
        print('Generating structure %d' % i)
        disp = dispmax * 2 * (0.5 - np.random.random((len(sym), 3)))
        positions = np.array(pos) + disp
        atoms = Atoms(''.join(sym), cell=cell, pbc=False, positions=positions)
        for j in range(len(sym)):
            dr = np.linalg.norm(positions[j] - pos[j])
            assert dr < (rc - d), (dr, rc - d)

        suffix = '"_no_repulsion.skf"'
        calc = DftbPlusCalc(atoms, maximum_angular_momenta=mam,
                            use_spline=False,
                            Hamiltonian_SlaterKosterFiles_Suffix=suffix)
        atoms.set_calculator(calc)
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        e_rep, f_rep = calculate_target_repulsion(atoms, rc)
        finalize(atoms, energy=e+e_rep, forces=f+f_rep, stress=None)
        db.write(atoms, relaxed=1)
    return


d, rc, rmin = 2., 2.5, 1.5
pair = 'Mg-O'
rcuts = {pair: rc, 'O-O': None, 'Mg-Mg': None}
mam = {'Mg': 0, 'O': 1}
dbfile = 'training.json'

if os.path.exists(dbfile):
    os.remove(dbfile)
generate_database(dbfile, d, rc)

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
                 kBT=2.0,
                 force_scaling=1.,
                 rcuts=rcuts,
                 rmins=rmins,
                 powers=powers)

    residual = calc.fit_repulsion([dbfile], run_checks=False)
    eps = 1e-6 if 'poly' in mode else 1e-2
    assert residual < eps, (mode, residual, eps)
    os.system('mv %s.pdf %s_%s.pdf' % (pair, pair, mode))
