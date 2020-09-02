import os
from random import random
import numpy as np
from time import time, sleep
from ase import Atoms
from ase.io import write, read
from ase.io.formats import UnknownFileTypeError
from ase.data import atomic_numbers
from ase.build import niggli_reduce
from ase.constraints import FixBondLengths, Filter
from ase.ga import get_raw_score, set_raw_score
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.population import Population
from ase.ga.utilities import (get_all_atom_types, closest_distances_generator,
                              atoms_too_close)
from ase.ga.offspring_creator import OperationSelector
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.bulk_utilities import CellBounds
from ase.ga.bulk_startgenerator import StartGenerator
from ase.ga.bulk_crossovers import CutAndSplicePairing
from ase.ga.bulk_mutations import *
from ase.ga.standardmutations import RattleMutation
from tango.relax_utils import push_apart, relax_precon, finalize
from tango.calculators import DftbPlusCalculator

comparator = OFPComparator(n_top=None, dE=1.0, cos_dist_max=5e-2,
                           rcut=10., binwidth=0.05, pbc=[True] * 3,
                           sigma=0.1, nsigma=4, recalculate=False)


def penalize(t):
    # penalize OH dissociation:
    raw_score = get_raw_score(t)
    tags = t.get_tags()
    for tag in list(set(tags)):
        indices = list(np.where(tags==tag)[0])
        if len(indices) != 2:
            continue
        d = t.get_distance(indices[0], indices[1], mic=True)
        if d > 1.2:
            raw_score -= 100.
    # penalize short O-O bonds:
    o_atoms = [atom.index for atom in t if atom.symbol == 'O']
    for o_atom in o_atoms:
        other_o_atoms = [i for i in o_atoms if i != o_atom]
        d = t.get_distances(o_atom, other_o_atoms, mic=True)
        if np.min(d) < 1.0:
            raw_score -= 1e9
    # penalize short H-H bonds:
    h_atoms = [atom.index for atom in t if atom.symbol == 'H']
    for h_atom in h_atoms:
        other_h_atoms = [i for i in h_atoms if i != h_atom]
        d = t.get_distances(h_atom, other_h_atoms, mic=True)
        if np.min(d) < 1.0:
            raw_score -= 1e9
    # penalize explosion:
    max_volume_per_atom = 50.
    if t.get_volume() / len(t) >= max_volume_per_atom:
        raw_score -= 1e9
    set_raw_score(t, raw_score)


def singlepoint(t, kptdensity=3.5):
    if get_raw_score(t) < -1e5:
        return t
    try:
        calc = DftbPlusCalculator(t, kpts=kptdensity, use_spline=True,
                              maximum_angular_momenta={'Pd': 2, 'H': 0, 'O': 1})
        t.set_calculator(calc)
        E = t.get_potential_energy()
        F = t.get_forces()
        S = t.get_stress()
        finalize(t, energy=E, forces=F, stress=S)
        penalize(t)
    except (IOError, TypeError, RuntimeError, UnboundLocalError) as err:
        print(err)
        print('Warning: problems with singlepoint recalculation')
        finalize(t, energy=1e9, forces=None, stress=None)
    return t


def relax_one(t, kptdensity=3.5):
    cellbounds = CellBounds(bounds={'phi': [0.1 * 180., 0.9 * 180.],
                                    'chi': [0.1 * 180., 0.9 * 180.],
                                    'psi': [0.1 * 180., 0.9 * 180.],
                                    'a': [1.5, 20], 'b': [1.5, 20],
                                    'c':[1.5, 20]})

    if not cellbounds.is_within_bounds(t.get_cell()):
        print('Candidate outside cellbounds -- skipping')
        finalize(t, energy=1e9, forces=None, stress=None)
        return t

    tags = t.get_tags()
    pos = t.get_positions()
    pairs = []
    for tag in list(set(tags)):
        indices = list(np.where(tags == tag)[0])
        if len(indices) == 2:
            pairs.append(indices)
    c = FixBondLengths(pairs)
    t.set_constraint(c)
    blmin = {(1, 1): 1.8, (1, 8): 0.9, (1, 46): 1.8, (8, 8): 2.0,
             (8, 46): 1.5, (46, 46): 1.5}
    t = push_apart(t, blmin, variable_cell=True)
    del t.constraints
    oh_bondlength = 0.97907
    for (o_atom, h_atom) in pairs:
        vec = t.get_distance(o_atom, h_atom, mic=True, vector=True)
        pos[h_atom] = pos[o_atom] + vec * oh_bondlength / np.linalg.norm(vec)
    t.set_positions(pos)

    print('Starting relaxation', flush=True)
    clock = time()
    t.wrap()
    calc = DftbPlusCalculator(t, kpts=0.66*kptdensity,
                              use_spline=True, read_chg=True,
                              maximum_angular_momenta={'Pd': 2, 'H': 0, 'O': 1})

    try:
        t = relax_precon(t, calc, fmax=2e-1, smax=1e-2, variable_cell=True,
                         optimizer='LBFGS', a_min=1e-4, cellbounds=cellbounds,
                         fix_bond_lengths_pairs=pairs, logfile='opt_first.log',
                         trajfile='opt_first.traj')
    except (IOError, TypeError, RuntimeError, UnboundLocalError) as err:
        # SCC or geometry optimization convergence problem
        print(err)
        if isinstance(t, Filter):
            t = t.atoms

    del t.constraints
    t.wrap()

    calc = DftbPlusCalculator(t, kpts=kptdensity,
                              use_spline=True, read_chg=True,
                              maximum_angular_momenta={'Pd': 2, 'H': 0, 'O': 1})

    try:
        t = relax_precon(t, calc, fmax=1e-1, smax=5e-3, variable_cell=True,
                         optimizer='LBFGS', a_min=1e-4, cellbounds=cellbounds,
                         fix_bond_lengths_pairs=pairs, logfile='opt.log',
                         trajfile='opt.traj')
    except (IOError, TypeError, RuntimeError, UnboundLocalError) as err:
        # SCC or geometry optimization convergence problem
        print(err)
        try:
            t = read('opt.traj@-1')
            energy = t.get_potential_energy()
            forces = t.get_forces()
            stress = t.get_stress()
        except (FileNotFoundError, UnknownFileTypeError) as err:
            print(err)
            energy, forces, stress = (1e9, None, None)

        if isinstance(t, Filter):
            t = t.atoms
        finalize(t, energy=energy, forces=forces, stress=stress)

    print('Relaxing took %.3f seconds.' % (time() - clock), flush=True)
    os.system('mv opt_first.traj prev_first.traj')
    os.system('mv opt_first.log prev_first.log')
    os.system('mv opt.log prev.log')
    os.system('mv opt.traj prev.traj')
    penalize(t)
    return t

def run_ga(n_to_test, kptdensity=3.5):
    population_size = 20
    da = DataConnection('godb.db')
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types, 0.05)  # 0.5
    blmin[(1, 8)] = blmin[(8, 1)] = 1.0

    # defining genetic operators:
    mutation_probability = 0.75
    pairing = CutAndSplicePairing(blmin, p1=1., p2=0., minfrac=0.15,
                                  use_tags=True)
    cellbounds = CellBounds(bounds={'phi': [0.2 * 180., 0.8 * 180.],
                                    'chi': [0.2 * 180., 0.8 * 180.],
                                    'psi': [0.2 * 180., 0.8 * 180.]})
    strainmut = StrainMutation(blmin, stddev=0.7, cellbounds=cellbounds,
                               use_tags=True)
    permut = PermutationMutation(blmin, probability=0.25, use_tags=True)

    blmin_soft = closest_distances_generator(all_atom_types, 0.1)
    softmut = SoftMutation(blmin_soft, bounds=[2., 5.],
                           fconstfunc=fconstfunc, use_tags=True)
    rotmut = RotationalMutation(blmin, fraction=0.3, min_angle=0.5 * np.pi)
    rattlemut = RattleMutation(blmin, n_to_optimize, rattle_prop=0.8,
                               rattle_strength=2.5, use_tags=True)
    rattlerotmut = RattleRotationalMutation(rattlemut, rotmut)
    mutations = OperationSelector([4., 4., 0., 4.],
                                  [softmut, strainmut, permut, rattlerotmut])

    if True:
        # recalculate raw scores
        structures = da.get_all_relaxed_candidates()
        for atoms in structures:
            atoms = singlepoint(atoms, kptdensity=kptdensity)
            da.c.delete([atoms.info['relax_id']])
            if 'data' not in atoms.info:
                atoms.info['data'] = {}
            da.add_relaxed_step(atoms)
        print('Finished recalculating raw scores')

    # relaxing the initial candidates:
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        a.wrap()
        a = relax_one(a, kptdensity=kptdensity)
        da.add_relaxed_step(a)

    # create the population
    population = Population(data_connection=da,
                            population_size=population_size,
                            comparator=comparator,
                            logfile='log.txt')

    current_pop = population.get_current_population()
    strainmut.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)
    pairing.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)

    # Test n_to_test new candidates
    ga_raw_scores = []
    step = 0
    for step in range(n_to_test):
        print('Starting configuration number %d' % step, flush=True)

        clock = time()
        a3 = None
        r = random()
        if r > mutation_probability:
            while a3 is None:
                a1, a2 = population.get_two_candidates()
                a3, desc = pairing.get_new_individual([a1, a2])
        else:
            while a3 is None:
                a1 = population.get_one_candidate()
                a3, desc = mutations.get_new_individual([a1])

        dt = time() - clock
        op = 'pairing' if r > mutation_probability else 'mutating'
        print('Time for %s candidate(s): %.3f' % (op, dt), flush=True)

        a3.wrap()
        da.add_unrelaxed_candidate(a3, description=desc)

        a3 = relax_one(a3, kptdensity=kptdensity)
        da.add_relaxed_step(a3)

        # Various updates:
        population.update()
        current_pop = population.get_current_population()

        if step % 10 == 0:
            strainmut.update_scaling_volume(current_pop, w_adapt=0.5,
                                            n_adapt=4)
            pairing.update_scaling_volume(current_pop, w_adapt=0.5, n_adapt=4)
            write('current_population.traj', current_pop)

        # Print out information for easy analysis/plotting afterwards:
        if r > mutation_probability:
            print('Step %d %s %.3f %.3f %.3f' % (step, desc,\
                   get_raw_score(a1), get_raw_score(a2), get_raw_score(a3)))
        else:
            print('Step %d %s %.3f %.3f' % (step, desc,\
                   get_raw_score(a1), get_raw_score(a3)))

        print('Step %d highest raw score in pop: %.3f' % \
              (step, get_raw_score(current_pop[0])))
        ga_raw_scores.append(get_raw_score(a3))
        print('Step %d highest raw score generated by GA: %.3f' % \
              (step, max(ga_raw_scores)))

    emin = population.pop[0].get_potential_energy()
    print('GA finished after step %d' % step)
    print('Lowest energy = %8.3f eV' % emin, flush=True)
    write('all_candidates.traj', da.get_all_relaxed_candidates())
    write('current_population.traj', population.get_current_population())


def prepare_ga(dbfile='godb.db', splits={(2,): 1}, N=20):

    blocks = [('Pd', 4), ('OH', 8)]  # the building blocks
    volume = 50. * 4 # volume in angstrom^3

    l = [list(Atoms(block).numbers)*count for block, count in blocks]
    stoichiometry = [item for sublist in l for item in sublist]
    atom_numbers = list(set(stoichiometry))

    blmin = closest_distances_generator(atom_numbers=atom_numbers,
                                        ratio_of_covalent_radii=0.6)
    blmin[(1, 8)] = blmin[(8, 1)] = 2.0

    cellbounds = CellBounds(bounds={'phi': [0.2 * 180., 0.8 * 180.],
                                    'chi': [0.2 * 180., 0.8 * 180.],
                                    'psi': [0.2 *180., 0.8 * 180.],
                                    'a': [2, 8], 'b': [2, 8], 'c': [2, 8]})

    # create the starting population
    sg = StartGenerator(blocks, blmin, volume, cellbounds=cellbounds,
                        splits=splits)

    # create the database to store information in
    da = PrepareDB(db_file_name=dbfile, stoichiometry=stoichiometry)

    for i in range(N):
        a = sg.get_new_candidate()
        a.set_initial_magnetic_moments(magmoms=None)
        niggli_reduce(a)
        da.add_unrelaxed_candidate(a)

    return
