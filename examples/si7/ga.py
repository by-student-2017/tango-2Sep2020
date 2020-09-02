'''
Here we define the methods used for the global optimization
runs using a genetic algorithm. If you're new to the GA
framework in ASE, it is recommended to also consult:
https://wiki.fysik.dtu.dk/ase/ase/ga.html
https://wiki.fysik.dtu.dk/ase/tutorials/ga/ga_optimize.html
'''
from random import random, seed
import numpy as np
from time import time
from ase import Atoms
from ase.io import write, read
from ase.data import atomic_numbers
from ase.ga import get_raw_score, set_raw_score
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.population import Population
from ase.ga.utilities import (get_all_atom_types, closest_distances_generator,
                              atoms_too_close)
from ase.ga.offspring_creator import OperationSelector
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.startgenerator import StartGenerator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.standardmutations import RattleMutation, MirrorMutation
from tango.relax_utils import push_apart, relax_precon, finalize
from tango.calculators import DftbPlusCalculator

# This is our OFPComparator instance which will be
# used to judge whether or not two structures are identical:
comparator = OFPComparator(n_top=None, dE=1.0, cos_dist_max=1e-3,
                           rcut=10., binwidth=0.05, pbc=[False] * 3,
                           sigma=0.1, nsigma=4, recalculate=False)


def singlepoint(t):
    ''' This function will be used in the iter007 step, which restarts
    from the runXXX/godb.db databases from the iter006 step. 
    Since the repulsive interactions have been reparametrized in
    between, the structures in the database need to be re-evaluated.
    '''
    calc = DftbPlusCalculator(t, kpts=(1, 1, 1), use_spline=True,
                              maximum_angular_momenta={'Si': 1})
    t.set_calculator(calc)
    E = t.get_potential_energy()
    F = t.get_forces()
    S = None
    finalize(t, energy=E, forces=F, stress=S)
    return t

def relax_one(t):
    ''' This method defines how to locally minimize a given
    atoms object 't'.
    '''
    # The provided structure will often contain atoms separated
    # by relatively short distances. Pushing these atoms a bit
    # apart using a soft potential will reduce the number of
    # subsequent ionic steps and will help avoid DFTB convergence
    # problems.
    pos = t.get_positions()
    numbers = list(set(t.get_atomic_numbers()))
    blmin = closest_distances_generator(numbers, 0.5)
    t = push_apart(t, blmin, variable_cell=False)

    print('Starting relaxation', flush=True)
    clock = time()
    t.wrap()

    # Define the DFTB+ calculator:
    calc = DftbPlusCalculator(t, kpts=(1, 1, 1), use_spline=True,
                              maximum_angular_momenta={'Si': 1})

    # Start the actual relaxation using the 
    # tango.relax_utils.relax_precon method.
    # This wraps around the preconditioned
    # optimizers in ASE and also takes care of
    # setting the raw score etc.:
    try:
        t = relax_precon(t, calc, fmax=1e-2, variable_cell=False,
                         logfile=None, trajfile=None)
    except IOError:
        # the DFTB+ ASE calculator throws an IOError
        # in case it couldn't find the 'results.tag'
        # output file, which may sporadically happen
        # due to SCC converge issues when e.g. a Si7
        # cluster is breaking into fragments. We simply
        # handle this by aborting the relaxation and
        # assigning a very high energy to the structure:
        finalize(t, energy=1e9, forces=None, stress=None)

    print('Relaxing took %.3f seconds.' % (time() - clock), flush=True)
    return t


def run_ga(n_to_test, kptdensity=None):
    ''' This method specifies how to run the GA once the
    initial random structures have been stored in godb.db.
    '''
    # Various initializations:
    population_size = 10
    da = DataConnection('godb.db')
    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    slab = da.get_slab()
    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
    blmin = closest_distances_generator(all_atom_types,
                                        ratio_of_covalent_radii=0.05)

    # Defining the mix of genetic operators:
    mutation_probability = 0.3333
    pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
    rattlemut = RattleMutation(blmin, n_to_optimize,
                               rattle_prop=0.8, rattle_strength=1.5)
    mirrormut = MirrorMutation(blmin, n_to_optimize)
    mutations = OperationSelector([1., 1.], [rattlemut, mirrormut])

    if True:
        # Recalculate raw scores of any relaxed candidates
        # present in the godb.db database (only applies to 
        # iter007).
        structures = da.get_all_relaxed_candidates()
        for atoms in structures:
            atoms = singlepoint(atoms)
            da.c.delete([atoms.info['relax_id']])
            if 'data' not in atoms.info:
                atoms.info['data'] = {}
            da.add_relaxed_step(atoms)
        print('Finished recalculating raw scores')

    # Relax the randomly generated initial candidates:
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        a.wrap()
        a = relax_one(a)
        da.add_relaxed_step(a)

    # Create the population
    population = Population(data_connection=da,
                            population_size=population_size,
                            comparator=comparator,
                            logfile='log.txt')
    current_pop = population.get_current_population()

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

        a3 = relax_one(a3)
        da.add_relaxed_step(a3)

        # Various updates:
        population.update()
        current_pop = population.get_current_population()
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


def prepare_ga(dbfile='godb.db', N=20):
    ''' This method creates a database with the desired number
    of randomly generated structures.
    '''
    blocks = [('Si', 7)]  # the building blocks
    l = [list(Atoms(block).numbers) * count for block, count in blocks]
    stoichiometry = [int(item) for sublist in l for item in sublist]
    atom_numbers = list(set(stoichiometry))

    # This dictionary will be used to check that the shortest
    # Si-Si distances are above a certain threshold 
    # (here 1.5 Angstrom):
    blmin = {(14, 14): 1.5}

    # This defines the cubic simulation cell:
    slab = Atoms('', positions=np.zeros((0, 3)), cell=[16] * 3)

    # This defines the smaller box in which the 
    # initial coordinates are allowed to vary
    density = 0.12  # in atoms per cubic Angstrom
    aspect_ratios = np.array([1.0, 1.0, 1.0])
    v = len(stoichiometry) / density
    l = np.cbrt(v / np.product(aspect_ratios))
    cell = np.identity(3) * aspect_ratios * l
    p0 = 0.5 * (np.diag(slab.get_cell() - cell))
    box = [p0, cell]

    # Seed the random number generators using the system time,
    # to ensure that no two runs produce the same results:
    np.random.seed()
    seed()

    # Generate the random structures and add them to the database:
    sg = StartGenerator(slab=slab, atom_numbers=stoichiometry,
                        closest_allowed_distances=blmin,
                        box_to_place_in=box)

    da = PrepareDB(db_file_name=dbfile,
                   simulation_cell=slab,
                   stoichiometry=stoichiometry)

    for i in range(N):
        a = sg.get_new_candidate()
        da.add_unrelaxed_candidate(a)

    return
