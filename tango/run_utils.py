import os
import tempfile
from time import time
import numpy as np
from ase import Atoms
from ase.io import Trajectory, read, write
from ase.data import atomic_numbers, covalent_radii
from ase.db import connect
from ase.calculators.calculator import (kptdensity2monkhorstpack,
                                        PropertyNotImplementedError)
from ase.optimize import BFGS, FIRE
from ase.optimize.precon import Exp
from ase.ga.utilities import closest_distances_generator
try:
    from ase.ga.bulk_utilities import CellBounds
except ImportError:
    print('Warning: Could not import CellBounds from ase.ga.bulk_utilities')
from tango.utilities import convert_array
from tango.relax_utils import (push_apart, finalize, PreconFIRE_My,
                               PreconLBFGS_My)


def get_kpts(atoms, kptdensity):
    ''' Returns a tuple corresponding to the appropriate
    Monkhorst-Pack mesh for the requested k-point density
    and Atoms object.

    atoms: an Atoms object
    kptdensity: the target k-point density in reciprocal
                Angstrom. If None, the Gamma-point only
                mesh is returned (i.e. (1 x 1 x 1)).
    '''
    if kptdensity is None:
        kpts = (1, 1, 1)
    else:
        kpts = tuple(kptdensity2monkhorstpack(atoms, even=False,
                                              kptdensity=kptdensity))
    return kpts


def run_atom(element, Calculator, maximum_angular_momenta=None):
    ''' Returns the total energy of an isolated atom.

    element: symbol of the element
    Calculator: a suitable DFT or DFTB calculator (see tango.main)
    '''
    atoms = Atoms(element, cell=[12.] * 3, positions=[[6.,6.,6.]], pbc=True)
    if maximum_angular_momenta is None:
        calc = Calculator(atoms, kpts=(1,1,1), run_type='atom')
    else:
        calc = Calculator(atoms, kpts=(1,1,1),
                          maximum_angular_momenta=maximum_angular_momenta)
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    calc.exit()
    print('%s Atomic energy of %s: %8.3f' % \
          (Calculator.__name__, element, energy))
    return energy


def run_dftbplus_singlepoint(args):
    ''' Returns the DFTB+ total energy and forces of an Atoms
    object (as well as a copy of the original object).
    Runs in a temporary directory which is afterwards deleted.

    args: tuple, for use with multiprocessing.
          It should contains the following items:
          * the Atoms object
          * a suitable DFTB+ calculator class (see tango.main).
          * the kptdensity to apply (units of 1/A)
          * whether to use the spline representation of the
            repulsive interactions (otherwise the polynomial
            form will be used)
          * a dictionary with maximum angular momenta for
            each element in the structure
          * a dictionary with the (DFTB) energies of the isolated atoms
          * the referencing scheme (see main.py)
    '''
    cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    atoms_original, DftbPlusCalculator, kptden, use_spline, mam, ae, ref = args
    atoms = atoms_original.copy()

    calc = DftbPlusCalculator(atoms, kpts=get_kpts(atoms, kptden),
                              use_spline=use_spline,
                              maximum_angular_momenta=mam)
    atoms.set_calculator(calc)

    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
    except (ValueError, IOError, UnboundLocalError) as err:
        # Normally this is caused either by DFTB+ convergence
        # problems or by a very high repulsive energy, printed
        # in the DFTB+ output as '***********'
        energy = None
        forces = None

    references = {'e_dftb_ref': [], 'f_dftb_ref': np.zeros((len(atoms), 3))}

    if ref == 'atomic':
        sym = atoms.get_chemical_symbols()
        references['e_dftb_ref'].extend([ae['%s_DFTB' % s] for s in sym])

    elif energy is not None:
        for indices in ref:
            a = atoms[indices]
            a.set_calculator(calc)
            try:
                e = a.get_potential_energy()
                f = a.get_forces()
            except IOError as err:
                # Probably DFTB+ convergence problems
                energy = None
                forces = None
                references = None
                break
            references['e_dftb_ref'].append(e)
            references['f_dftb_ref'][indices] = f

    calc.exit()
    os.chdir(cwd)
    os.system('rm -r %s' % tmpdir)
    return (atoms_original, energy, forces, references)


def run_calc(dbfile, Calculator, kptdensity=None, relax=False, vc_relax=False,
             precon=True, maxsteps=20, maximum_angular_momenta=None,
             atomic_energies={}, referencing='atomic'):
    ''' Runs a calculator on each unrelaxed candidate in a database.

    dbfile: name of the database
    Calculator: a suitable calculator DFT or DFTB calculator class 
                (see tango.main)
    kptdensity: the k-point density to apply (in 1/A)
    relax: whether to do a short relaxation or only 
           perform a single-point calculation
    vc_relax: if also the cell vectors are to be varied
    precon: whether to use the preconditioned optimizers
    maxsteps: maximum number of ionic steps for the local optimizer
    maximum_angular_momenta: a dictionary with maximum angular momenta 
             for each element in the structure for DFTB calculations
    atomic_energies: dictionary with the DFT energies of the isolated
                     atoms. Used to calculate the reference energies
                     in the 'atomic' referencing scheme.
    referencing: the referencing scheme (see main.py)
    '''
    if vc_relax:
        assert precon

    db = connect(dbfile)
    relaxed_ids = set([row.gaid for row in db.select(relaxed=1)])

    for row in db.select(relaxed=0):
        if row.gaid in relaxed_ids:
            continue

        atoms = row.toatoms()
        mp = get_kpts(atoms, kptdensity)
        if maximum_angular_momenta is None:
            calc = Calculator(atoms, kpts=mp)
        else:
            calc = Calculator(atoms, kpts=mp,
                              maximum_angular_momenta=maximum_angular_momenta)
        atoms.set_calculator(calc)

        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        try:
            S = atoms.get_stress() 
        except PropertyNotImplementedError:
            S = None
        finalize(atoms, energy=E, forces=F, stress=S)

        relax = relax and maxsteps > 0
        if relax:
            atoms2 = atoms.copy()
            numbers = list(set(atoms2.get_atomic_numbers()))
            blmin = closest_distances_generator(numbers, 0.5)
            atoms2 = push_apart(atoms2, blmin)
            atoms2.set_calculator(calc)
            atoms2 = do_short_relax(atoms2, index=row.gaid,
                                    precon=precon, vc_relax=vc_relax, 
                                    maxsteps=maxsteps)
            if vc_relax:
                # Additional single-point run
                calc.exit()
                mp = get_kpts(atoms2, kptdensity)
                if maximum_angular_momenta is None:
                    calc = Calculator(atoms2, kpts=mp)
                else:
                    calc = Calculator(atoms2, kpts=mp,
                             maximum_angular_momenta=maximum_angular_momenta)
                atoms2.set_calculator(calc)

            E = atoms2.get_potential_energy()
            F = atoms2.get_forces()
            try:
                S = atoms2.get_stress()
            except PropertyNotImplementedError:
                S = None
            finalize(atoms2, energy=E, forces=F, stress=S)

        # Calculate energy and force references
        for a in [atoms] if not relax else [atoms, atoms2]:
            e_ref = []
            f_ref = np.zeros((len(atoms), 3))
            if referencing == 'atomic':
                sym = a.get_chemical_symbols()
                e_ref.extend([atomic_energies['%s_DFT' % s] for s in sym])
            else:
                for indices in referencing:
                    b = a[indices]
                    b.set_calculator(calc)
                    e_ref.append(b.get_potential_energy())
                    f_ref[indices] = b.get_forces()

            a.info['key_value_pairs']['e_dft_ref'] = convert_array(e_ref)
            a.info['key_value_pairs']['f_dft_ref'] = convert_array(f_ref)
            # Add the structure to the database:
            db.write(a, relaxed=1, gaid=row.gaid, **a.info['key_value_pairs'])
        calc.exit()
    return


def do_short_relax(atoms, index=None, vc_relax=False, precon=True,
                   maxsteps=20):
    ''' Performs a (usually short) local optimization.

    atoms: an Atoms object 
    index: index to be used as suffix for the output files
    vc_relax: whether to also optimize the cell vectors
              (after having run several steps with fixed cell)
    precon: whether to use the preconditioned optimizers
    maxsteps: maximum number of ionic steps
    '''
    if vc_relax:
        assert precon

    t = time()
    label = 'opt' if index is None else 'opt_' + str(index)
    logfile = '%s.log' % label
    trajfile = '%s.traj' % label

    traj = Trajectory(trajfile, 'a', atoms)
    nsteps = 0
    maxsteps_no_vc = maxsteps / 2 if vc_relax else maxsteps
    fmax = 2. if vc_relax else 0.1

    try:
        if precon:
            dyn = PreconLBFGS_My(atoms, precon=Exp(A=3), variable_cell=False,
                                 use_armijo=True, a_min=1e-2, logfile=logfile)
        else:
            dyn = BFGS(atoms, maxstep=0.4, logfile=logfile)
        dyn.attach(traj)
        dyn.run(fmax=fmax, steps=maxsteps_no_vc)
    except RuntimeError:
        nsteps += dyn.get_number_of_steps()
        if precon:
            dyn = PreconFIRE_My(atoms, precon=Exp(A=3), variable_cell=False,
                                use_armijo=False, logfile=logfile,
                                dt=0.1, maxmove=0.5, dtmax=1.0, finc=1.1)
        else:
            dyn = FIRE(atoms, logfile=logfile, dt=0.1, maxmove=0.5, dtmax=1.0,
                       finc=1.1)
        dyn.attach(traj)
        steps = maxsteps_no_vc - nsteps
        dyn.run(fmax=fmax, steps=steps)

    nsteps += dyn.get_number_of_steps()

    if vc_relax:
        L = atoms.get_volume() / 4.  # largest cell vector length allowed
        cellbounds = CellBounds(bounds={'phi':[20., 160.], 'a':[1.5, L],
                                        'chi':[20., 160.], 'b':[1.5, L],
                                        'psi':[20., 160.], 'c':[1.5, L]})
        try:
            dyn = PreconLBFGS_My(atoms, precon=Exp(A=3), variable_cell=True,
                                 use_armijo=True, logfile=logfile,
                                 cellbounds=cellbounds, a_min=1e-2)
            dyn.e1 = None
            try:
                dyn._just_reset_hessian
            except AttributeError:
                dyn._just_reset_hessian = True
            dyn.attach(traj)
            steps = maxsteps - nsteps
            dyn.run(fmax=0., smax=0., steps=steps)
        except RuntimeError:
            nsteps += dyn.get_number_of_steps()
            dyn = PreconFIRE_My(atoms, precon=Exp(A=3), variable_cell=True,
                                use_armijo=False, logfile=logfile,
                                cellbounds=cellbounds,
                                dt=0.1, maxmove=0.5, dtmax=1.0, finc=1.1)
            dyn.attach(traj)
            steps = maxsteps - nsteps
            dyn.run(fmax=0., steps=steps)

    name = atoms.calc.name
    print('%s relaxation took %.3f seconds' % (name, time()-t))
    return atoms


def run_dimers(dbfile, DftCalc, atomic_energies, element1, element2,
               minfrac=0.15, maxfrac=0.7, stepfrac=0.05, minE=5.):
    ''' Adds a dimer curve (with DFT energies and forces) for
    the given element pair to a database.

    dbfile: name of the database
    DftCalc: suitable DFT calculator class (see tango.main)
    atomic_energies: reference DFT energies of the separate atoms
    element1: symbol of the first element 
    element2: symbol of the second element
    minfrac, maxfrac, stepfrac: specifies the minimal and maximal
         dimer distances (and the spacing) in units of the sum of
         covalent radii
    minE: dimer distances where the dimer energy is less than minE
          above the sum of the reference energies are omitted
    '''
    positions = np.array([[6.] * 3, [8., 6., 6.]])
    atoms = Atoms(element1 + element2, cell=[12.] * 3,
                  positions=positions, pbc=True)

    calc = DftCalc(atoms, kpts=(1,1,1), run_type='dimer')

    db = connect(dbfile)
    num1 = atomic_numbers[element1]
    num2 = atomic_numbers[element2]
    e_ref = [atomic_energies['%s_DFT' % e] for e in [element1, element2]]
    crsum = covalent_radii[num1] + covalent_radii[num2]
    r = minfrac * crsum

    while True:
        if r * 1. / crsum > maxfrac:
            break

        positions[1, 0] = 6. + r
        atoms.set_positions(positions)
        atoms.set_calculator(calc)
        E = atoms.get_potential_energy()
        F = atoms.get_forces()

        if E - sum(e_ref) > minE:
            atoms.info['key_value_pairs'] = {}
            atoms.info['key_value_pairs']['e_dft_ref'] = convert_array(e_ref)
            atoms.info['key_value_pairs']['f_dft_ref'] = 0.
            finalize(atoms, energy=E, forces=F)
            db.write(atoms, r=r, **atoms.info['key_value_pairs'])
            r += stepfrac * crsum
        else:
            break

    calc.exit()
    return
