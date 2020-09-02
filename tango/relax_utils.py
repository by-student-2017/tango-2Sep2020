import os
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, Trajectory
from ase.geometry import find_mic, wrap_positions
from ase.constraints import (Filter, FixBondLengths,
                             StrainFilter, UnitCellFilter,
                             voigt_6_to_full_3x3_stress,
                             full_3x3_to_voigt_6_stress)
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase import optimize
from ase.optimize import BFGS, BFGSLineSearch, FIRE
from ase.optimize.precon import PreconLBFGS, PreconFIRE, Exp
from ase.ga import set_raw_score


class UnitCellFilterFixBondLengths(UnitCellFilter):
    """ Modification of ase.constraints.UnitCellFilter to allow
    fixing bond lengths during variable-cell relaxation.
    """
    def __init__(self, atoms, pairs, bondlengths=None, tolerance=1e-13,
                 maxiter=500, **kwargs):
        UnitCellFilter.__init__(self, atoms, **kwargs)
        self.pairs = np.asarray(pairs)
        self.bondlengths = bondlengths
        self.tolerance = tolerance
        self.maxiter = maxiter

        if self.bondlengths is None:
            self.bondlengths = np.zeros(len(self.pairs))
            for i, ab in enumerate(self.pairs):
                self.bondlengths[i] = atoms.get_distance(ab[0], ab[1], mic=True)

    def set_positions(self, new, **kwargs):
        # First, adjust positions (due to fixed bond lengths):
        old = self.atoms.get_positions()
        oldcell = self.atoms.get_cell()
        masses = self.atoms.get_masses()
        for i in range(self.maxiter):
            converged = True
            for j, ab in enumerate(self.pairs):
                a = ab[0]
                b = ab[1]
                cd = self.bondlengths[j]
                r0 = old[a] - old[b]
                d0 = find_mic([r0], oldcell, self.atoms._pbc)[0][0]
                d1 = new[a] - new[b] - r0 + d0
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = 0.5 * (cd**2 - np.dot(d1, d1)) / np.dot(d0, d1)
                if abs(x) > self.tolerance or np.isnan(x) or np.isinf(x):
                    new[a] += x * m / masses[a] * d0
                    new[b] -= x * m / masses[b] * d0
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')

        natoms = len(self.atoms)
        self.deform_grad = new[natoms:] / self.cell_factor
        current_cell = self.atoms.get_cell()
        new_cell = np.dot(self.orig_cell, self.deform_grad.T)
        scaled_pos = np.dot(new[:natoms], np.linalg.inv(current_cell))
        self.atom_positions[:] = new[:natoms]
        self.atoms.set_positions(np.dot(scaled_pos, new_cell), **kwargs)
        self.atoms.set_positions(self.atom_positions, **kwargs)
        self.atom_positions = self.atoms.get_positions()  # obsolete?
        self.atoms.set_cell(new_cell, scale_atoms=False)

    def get_forces(self, apply_constraint=False):
        atoms_forces = self.atoms.get_forces()

        # Now, adjust forces:
        constraint_forces = -atoms_forces
        old = self.atoms.get_positions()
        oldcell = self.atoms.get_cell()
        masses = self.atoms.get_masses()
        for i in range(self.maxiter):
            converged = True
            for j, ab in enumerate(self.pairs):
                a = ab[0]
                b = ab[1]
                cd = self.bondlengths[j]
                d = old[a] - old[b]
                d = find_mic([d], oldcell, self.atoms._pbc)[0][0]
                dv = atoms_forces[a] / masses[a] - atoms_forces[b] / masses[b]
                m = 1 / (1 / masses[a] + 1 / masses[b])
                x = -np.dot(dv, d) / cd ** 2
                if abs(x) > self.tolerance or np.isnan(x) or np.isinf(x):
                    atoms_forces[a] += x * m * d
                    atoms_forces[b] -= x * m * d
                    converged = False
            if converged:
                break
        else:
            raise RuntimeError('Did not converge')
        constraint_forces += atoms_forces

        stress = self.atoms.get_stress()
        volume = self.atoms.get_volume()
        virial = -volume * voigt_6_to_full_3x3_stress(stress)
        atoms_forces = np.dot(atoms_forces, self.deform_grad)
        dg_inv = np.linalg.inv(self.deform_grad)
        virial = np.dot(virial, dg_inv.T)

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask != 1.0).any():
            virial *= self.mask

        if self.constant_volume:
            vtr = virial.trace()
            np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

        natoms = len(self.atoms)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = virial / self.cell_factor

        self.stress = -full_3x3_to_voigt_6_stress(virial)/volume
        return forces


class VarianceError(object):
    """ Like ase.ga.relax_attaches.VariansBreak, but raises RuntimeError. """
    def __init__(self, atoms, dyn, min_stdev=0.05, N=15):
        self.atoms = atoms
        self.dyn = dyn
        self.N = N
        self.forces = []
        self.min_stdev = min_stdev

    def write(self):
        """ The method called by the optimizer in each step. """
        if len(self.forces) >= self.N:
            self.forces.pop(0)
        fmax = (self.atoms.get_forces() ** 2).sum(axis=1).max() ** 0.5
        self.forces.append(fmax)

        m = sum(self.forces) / float(len(self.forces))

        stdev = np.sqrt(sum([(c - m) ** 2 for c in self.forces]) /
                        float(len(self.forces)))

        if len(self.forces) >= self.N and stdev < self.min_stdev:
            raise RuntimeError('Stagnating optimization!')


class DivergenceError(object):
    """ Similar to VarianceError, but for diverging optimization. """
    def __init__(self, atoms, dyn, max_slope=0.01, N=5):
        self.atoms = atoms
        self.dyn = dyn
        self.max_slope = max_slope
        self.N = N
        self.energies = []

    def write(self):
        """ The method called by the optimizer in each step. """
        if len(self.energies) >= self.N:
            self.energies.pop(0)
        self.energies.append(self.atoms.get_potential_energy())

        if len(self.energies) >= self.N:
            x = np.array(range(len(self.energies)))
            y = np.array(self.energies)
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intersect = np.linalg.lstsq(A, y, rcond=None)[0]
            if slope > self.max_slope:
                raise RuntimeError('Diverging optimization!')


class VolumeError(object): 
    """ Raises RuntimeError when volume exceeds a given limit. """
    def __init__(self, atoms, dyn, max_volume):
        self.atoms = atoms
        self.dyn = dyn
        self.max_volume = max_volume

    def write(self):
        """ The method called by the optimizer in each step. """
        volume = self.atoms.get_volume()
        if volume > self.max_volume:
            raise RuntimeError('Maximal volume exceeded!')


class PreconLBFGS_My(PreconLBFGS):
    """ Modification of ase.optimize.precon.PreconLBFGS. """
    def __init__(self, atoms, cellbounds=None, **kwargs):
        self.cellbounds = cellbounds
        PreconLBFGS.__init__(self, atoms, **kwargs)

    def accept_step(self, x):
        if isinstance(self.atoms, UnitCellFilter):
            if self.cellbounds is not None:
                deform_grad = x.reshape(-1,3)[-3:] / self.atoms.cell_factor
                new_cell = np.dot(self.atoms.orig_cell, deform_grad.T)
                if not self.cellbounds.is_within_bounds(new_cell):
                    return False
        return True

    def func(self, x):
        """ Check cell shape to avoid the occasional crazy
        cells attempted by the optimizer.
        """
        if not self.accept_step(x):
            return 1e64
        return PreconLBFGS.func(self, x)

    def fprime(self, x):
        """ Gradient of the objective function for use of the optimizers. """
        if not self.accept_step(x):
            return 1e64*np.ones_like(x)
        return PreconLBFGS.fprime(self, x)


class PreconFIRE_My(PreconFIRE):
    """ Modification of ase.optimize.precon.PreconFIRE. """
    def __init__(self, atoms, cellbounds=None, **kwargs):
        self.cellbounds = cellbounds
        PreconFIRE.__init__(self, atoms, **kwargs)

    def accept_step(self, x):
        if isinstance(self.atoms, UnitCellFilter):
            if self.cellbounds is not None:
                deform_grad = x.reshape(-1,3)[-3:] / self.atoms.cell_factor
                new_cell = np.dot(self.atoms.orig_cell, deform_grad.T)
                if not self.cellbounds.is_within_bounds(new_cell):
                    return False
        return True

    def func(self, x):
        """ Check cell shape to avoid the occasional crazy
        cells attempted by the optimizer.
        """
        if not self.accept_step(x):
            return 1e64
        return PreconFIRE.func(self, x)


def finalize(atoms, energy=None, forces=None, stress=None):
    """ Saves attributes by attaching a SinglePointCalculator
    and sets the default raw score (equal to the negative
    of the potential energy).
    """
    atoms.wrap()
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                 stress=stress)
    atoms.set_calculator(calc)
    raw_score = -atoms.get_potential_energy()
    set_raw_score(atoms, raw_score)


def relax_precon(atoms, calc, fmax=5e-2, smax=1e-4, variable_cell=False,
                 trajfile=None, logfile=None, maxsteps=10000, verbose=True,
                 dEmin=1e-3, a_min=1e-8, optimizer='LBFGS', cellbounds=None,
                 fix_bond_lengths_pairs=None, debug=False):
    """ Locally optimizes the geometry using the preconditioned optimizers.
    
    atoms: the Atoms object
    calc: a calculator instance to attach to the Atoms object
    fmax: the force convergence criterion (in eV/Angstrom)
    smax: the stress convergence criterion (in eV/Angstrom^3)
    variable_cell: whether to also optimize the cell vectors
    trajfile: filename of the trajectory to attach to the optimizers
    logfile: filename of the logfile for the optimizers
    maxsteps: maximum allowed total number of ionic steps
    verbose: whether to print output or remain silent
    dEmin: minimum (absolute) energy difference for the main loop
           iterations; if the energy difference is smaller, the
           minimization is aborted
    a_min: minimal value for the linesearch parameter in the
           PreconLBFGS optimizer
    optimizer: 'LBFGS' or 'FIRE', the preferred type of
               (preconditioned) optimizer
    cellbounds: ase.ga.bulk_utilities.CellBounds instance for
                setting bounds on the cell shape and size
                during variable-cell relaxation
    fix_bond_lengths_pairs: list of (i,j) tuples of atom pairs
                for which the (minimum image) bond distance
                will be kept fixed
    debug: if True, extra output may be printed to stdout
    """
    assert optimizer in ['LBFGS', 'FIRE']
    
    class UnitCellFilterFixBondLengthsWrapper(UnitCellFilterFixBondLengths):
        """ Wrapper to nitCellFilterFixBondLengths,
        required because of ase.optimize.precon.estimate_mu.
        """
        def __init__(self, atoms):
            UnitCellFilterFixBondLengths.__init__(self, atoms, maxiter=1000,
                                                  pairs=fix_bond_lengths_pairs)

    if variable_cell:
        atoms.set_pbc(True)
    else:
        smax = None
        cellbounds = None

    atoms.wrap()
    atoms.set_calculator(calc)
    nsteps = 0

    if trajfile is not None:
        if os.path.exists(trajfile):
            os.remove(trajfile)
        traj = Trajectory(trajfile, 'a', atoms)
    else:
        traj = None

    if fix_bond_lengths_pairs is not None:
        original_constraints = [c for c in atoms.constraints]
        bond_lengths = np.zeros(len(fix_bond_lengths_pairs))
        for i, ab in enumerate(fix_bond_lengths_pairs):
            bond_lengths[i] = atoms.get_distance(ab[0], ab[1], mic=True)
        fbl_constraint = FixBondLengths(fix_bond_lengths_pairs)
        atoms.set_constraint(original_constraints + [fbl_constraint])

    if variable_cell:
        # Start with partial relaxation using fixed cell vectors
        if fix_bond_lengths_pairs or optimizer == 'FIRE':
            # PreconFIRE is more suitable in this case
            dyn = PreconFIRE_My(atoms, precon=Exp(A=3), use_armijo=True,
                                variable_cell=False, logfile=logfile)
        else:
            dyn = PreconLBFGS_My(atoms, precon=Exp(A=3), use_armijo=True,
                                 variable_cell=False, logfile=logfile,
                                 maxstep=0.2, a_min=a_min)
        attach1 = VarianceError(atoms, dyn, N=5)
        dyn.attach(attach1)
        attach2 = DivergenceError(atoms, dyn, N=2, max_slope=0.05)
        dyn.attach(attach2)
        if traj is not None:
            dyn.attach(traj)
        try:
            dyn.run(fmax=1., steps=25)
        except RuntimeError as err:
            if debug:
                print(err.message, flush=True)
        nsteps += dyn.get_number_of_steps()

    if variable_cell and fix_bond_lengths_pairs is not None:
        # Fixing bond lengths will be taken care of
        # by the modified UnitCellFilter 
        atoms.set_constraint(original_constraints)
        atoms = UnitCellFilterFixBondLengthsWrapper(atoms)
    elif variable_cell:
        atoms = UnitCellFilter(atoms)

    steps = 30
    niter = 0
    maxiter = 50
    nerr = 0
    maxerr = 10
    e_prev = 1e64

    while nsteps < maxsteps and niter < maxiter and nerr < maxerr:
        e = atoms.get_potential_energy()
        if abs(e - e_prev) < dEmin:
            break
        e_prev = e

        try:
            if optimizer == 'LBFGS':
                dyn = PreconLBFGS_My(atoms, precon=Exp(A=3), use_armijo=True,
                                     logfile=logfile, a_min=a_min,
                                     variable_cell=False, maxstep=0.2,
                                     cellbounds=cellbounds)
            elif optimizer == 'FIRE':
                dyn = PreconFIRE_My(atoms, precon=Exp(A=3), use_armijo=True,
                                    variable_cell=False, logfile=logfile,
                                    maxmove=0.5, cellbounds=cellbounds)
            dyn.e1 = None
            try:
                dyn._just_reset_hessian
            except AttributeError:
                dyn._just_reset_hessian = True
            if traj is not None:
                dyn.attach(traj)
            attach1 = VarianceError(atoms, dyn, N=5)
            dyn.attach(attach1)
            attach2 = DivergenceError(atoms, dyn, N=2, max_slope=0.05)
            dyn.attach(attach2)
            dyn.run(fmax=fmax, smax=smax, steps=steps)
            nsteps += dyn.get_number_of_steps()
        except RuntimeError as err:
            if debug:
                print(err.message, flush=True)
            nerr += 1
            nsteps += dyn.get_number_of_steps()
            dyn = PreconFIRE_My(atoms, precon=Exp(A=3), use_armijo=False,
                                variable_cell=False, logfile=logfile,
                                maxmove=0.5, cellbounds=cellbounds)
            if traj is not None:
                dyn.attach(traj)
            attach1 = VarianceError(atoms, dyn, N=5)
            dyn.attach(attach1)
            attach2 = DivergenceError(atoms, dyn, N=2, max_slope=0.05)
            dyn.attach(attach2)
            try: 
                dyn.run(fmax=fmax, smax=smax, steps=steps)
            except RuntimeError as err:
                if debug:
                    print(err.message, flush=True)
                nerr += 1
            nsteps += dyn.get_number_of_steps()

        niter += 1

        try:
            if dyn.converged():
                break
        except RuntimeError:
            break

        if isinstance(atoms, UnitCellFilterFixBondLengthsWrapper):
            atoms = atoms.atoms
            atoms.wrap()
            pos = atoms.get_positions()
            cell = atoms.get_cell()
            pbc = atoms.get_pbc()
            for (i1, i2), bl in zip(fix_bond_lengths_pairs, bond_lengths):
                vec = pos[i2] - pos[i1]
                vec = find_mic([vec], cell, pbc)[0][0]
                pos[i2] = pos[i1] + vec * bl / np.linalg.norm(vec)
            atoms.set_positions(pos)
            atoms = UnitCellFilterFixBondLengthsWrapper(atoms)

    calculate_regular = False

    if fix_bond_lengths_pairs is not None:
        # Guarantee that the fixed bonds have the specified length
        pos = atoms.get_positions()

        if isinstance(atoms, Filter):
            cell = atoms.atoms.get_cell()
            pbc = atoms.atoms.get_pbc() 
        else:
            cell = atoms.get_cell()
            pbc = atoms.get_pbc()

        for (i1, i2), bl in zip(fix_bond_lengths_pairs, bond_lengths):
            vec = pos[i2] - pos[i1]
            vec = find_mic([vec], cell, pbc)[0][0]
            pos[i2] = pos[i1] + vec * bl / np.linalg.norm(vec)

        if isinstance(atoms, Filter):
            atoms = atoms.atoms
            atoms.set_positions(pos[:len(atoms)])
            atoms = UnitCellFilterFixBondLengthsWrapper(atoms)
            try:
                E = atoms.get_potential_energy()
                F = atoms.get_forces()[:len(atoms.atoms)]
                S = atoms.stress
            except RuntimeError:
                calculate_regular = True
        else:
            atoms.set_constraint(original_constraints)
            atoms.set_positions(pos)
            atoms.set_constraint(original_constraints + [fbl_constraint])
            calculate_regular = True

    if isinstance(atoms, Filter):
        atoms = atoms.atoms

    if fix_bond_lengths_pairs is None or calculate_regular:
        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        S = atoms.get_stress() if variable_cell else None

    atoms.wrap()

    if verbose:
        print('Done E=%8.3f, maxF=%5.3f, maxS=%5.3e, nsteps=%d' % \
              (E, (F**2).sum(axis=1).max()**0.5,
               0. if S is None else np.max(np.abs(S)), nsteps), flush=True)

    finalize(atoms, energy=E, forces=F, stress=S)
    return atoms


def is_converged(a, fmax=None, smax=None):
    """ Returns whether the force and stress criteria are met. """
    force_ok = True
    if fmax is not None:
        force_ok = (a.get_forces() ** 2).sum(axis=1).max() < fmax ** 2
    stress_ok = True
    if smax is not None:
        stress_ok = (a.get_stress() ** 2).max() < smax ** 2
    return force_ok and stress_ok


def relax_standard(atoms, calc, fmax=5e-2, smax=1e-4, variable_cell=False,
                   trajfile=None, logfile=None, maxsteps=10000,
                   verbose=False, dEmin=1e-3, optimizer='BFGSLineSearch',
                   ucf=False):
    """ Locally optimizes the geometry using the non-preconditioned
    optimizers. Cell optimization is done by alternating
    between fixed-cell and variable-cell relaxations.
    
    atoms: the Atoms object
    calc: a calculator instance to attach to the Atoms object
    fmax: the force convergence criterion (in eV/Angstrom)
    smax: the stress convergence criterion (in eV/Angstrom^3)
    variable_cell: whether to also optimize the cell vectors
    trajfile: filename of the trajectory to attach to the optimizers
    logfile: filename of the logfile for the optimizers
    maxsteps: maximum allowed total number of ionic steps
    verbose: whether to print output or remain silent
    dEmin: minimum (absolute) energy difference for the main loop
           iterations; if the energy difference is smaller, the
           minimization is aborted
    optimizer: name of the ASE optimizer
    ucf: whether to use a UnitCellFilter (True) or StrainFilter(False)
         during the variable-cell relaxations
    """
    if variable_cell:
        atoms.set_pbc(True)
    else:
        smax = None

    atoms.wrap()
    atoms.set_calculator(calc)
    nsteps = 0

    if trajfile is not None:
        if os.path.exists(trajfile):
            os.remove(trajfile)
        traj = Trajectory(trajfile, 'a', atoms)
    else:
        traj = None

    niter = 0
    maxiter = 100
    steps = 50 if variable_cell else 125
    while nsteps < maxsteps and niter < maxiter:
        try:
            if optimizer in ['BFGS', 'BFGSLineSearch', 'LBFGSLineSearch']:
                dyn = getattr(optimize, optimizer)(atoms, logfile=logfile,
                                                   maxstep=0.2)
            elif optimizer == 'FIRE': 
                dyn = optimize.FIRE(atoms, logfile=logfile, dt=0.025,
                                    finc=1.25, dtmax=0.3, maxmove=0.2)
            if traj is not None: 
                dyn.attach(traj)
            vb = VarianceError(atoms, dyn)
            dyn.attach(vb)
            dyn.run(fmax=fmax, steps=steps)
            nsteps += dyn.get_number_of_steps()
        except (RuntimeError, np.linalg.linalg.LinAlgError) as err:
            # Sometimes, BFGS can fail due to
            # numpy.linalg.linalg.LinAlgError:
            # Eigenvalues did not converge
            nsteps += dyn.get_number_of_steps()
            dyn = optimize.FIRE(atoms, logfile=logfile, dt=0.05,
                                finc=1.25, dtmax=0.3, maxmove=0.2)
            if traj is not None: 
                dyn.attach(traj)
            vb = VarianceError(atoms, dyn)
            dyn.attach(vb)
            try:
                dyn.run(fmax=fmax, steps=steps)
            except RuntimeError:
                pass
            nsteps += dyn.get_number_of_steps()

        if variable_cell:
            filt = UnitCellFilter(atoms) if ucf else StrainFilter(atoms)
            dyn = optimize.MDMin(filt, dt=0.05, logfile=logfile)
            if traj is not None:
                dyn.attach(traj)
            dyn.run(fmax=fmax, steps=5)
            nsteps += dyn.get_number_of_steps()

        niter += 1

        if is_converged(atoms, fmax=fmax, smax=smax):
            break

    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    S = atoms.get_stress() if variable_cell else None

    if verbose:
        print('Done E=%8.3f, maxF=%5.3f, maxS=%5.3e, ncalls=%d'% \
              (E, (F ** 2).sum(axis=1).max() ** 0.5,
               0. if S is None else np.max(np.abs(S)), nsteps))

    finalize(atoms, energy=E, forces=F, stress=S)
    return atoms


def push_apart(atoms, blmin, variable_cell=False, maxsteps=500,
               logfile=None, trajectory=None):
    """ Push atoms apart so as to (try) satisfy the blmin dictionary
    of minimal interatomic distances while not displacing the
    atoms too much. The default SHPP calculator is used for
    this purpose.

    atoms: an Atoms object
    blmin: dictionary with the minimal interatomic distance
           for each (sorted) pair of atomic numbers
    variable_cell: whether to allow the cell vectors to vary
    maxsteps: maximum number of optimizer steps
    trajectory: (filename of the) trajectory to attach to the optimizer
    logfile: (filename of the) logfile for the optimizer
    """
    if variable_cell:
        blminmax = max([blmin[k] for k in blmin if k[0]==k[1]])
        cell = atoms.get_cell()
        for i in range(3):
            cell[i] *= max(1, blminmax / np.linalg.norm(cell[i]))
        atoms.set_cell(cell, scale_atoms=True)

    calc = SHPP(atoms, blmin)
    atoms.set_calculator(calc)
    dyn = BFGS(atoms, maxstep=0.05, logfile=logfile, trajectory=trajectory)
    dyn.run(fmax=0.05, steps=maxsteps)
    return atoms


class SHPP(Calculator):
    """ Soft+Harmonic Pair Potential -- a combination
    of LAMMPS-pair_style 'soft' pair potential
    plus restorative forces.

    The total energy is given by:
        E_tot(R) = E_harm(R) + E_soft(R)
        E_harm(R) = Sum_i 0.5*k*|R_i-R_i,start|^2
        E_soft(R) = Sum_i<j A*(1+cos(|R_i-R_j|*pi/d_min))

    atoms: an Atoms object
    blmin: dictionary with the minimal interatomic distance
           for each (sorted) pair of atomic numbers
    k: spring constant for the harmonic potential in eV/Angstrom^2
    A: prefactor for the 'soft' potential in eV
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms, blmin, k=0.5, A=10.):
        Calculator.__init__(self)
        self.positions = atoms.get_positions()
        self.blmin = blmin
        self.k = k
        self.A = A
        self.N = len(atoms)
        rcut = max(self.blmin.values())
        self.nl = NeighborList([rcut / 2.] * self.N, skin=1., bothways=True,
                               self_interaction=False)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = 0, np.zeros((self.N, 3))
        e, f = self._calculate_harm(atoms)
        energy += e
        forces += f
        e, f = self._calculate_soft(atoms)
        energy += e
        forces += f
        self.results = {'energy': energy, 'forces': forces}

    def _calculate_harm(self, atoms):
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        vectors = atoms.get_positions() - self.positions
        e = 0.
        f = np.zeros((self.N, 3))
        for i,v in enumerate(vectors):
            v, d = find_mic([v], cell, pbc)
            e += 0.5 * self.k * (d ** 2)
            f[i] = -self.k * v
        return e, f

    def _calculate_soft(self, atoms):
        cell = atoms.get_cell()
        num = atoms.get_atomic_numbers()
        pos = atoms.get_positions()
        self.nl.update(atoms)
        e = 0
        f = np.zeros((self.N, 3))
        for i in range(self.N):
            indices, offsets = self.nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])
            v = p - pos[i]
            for j, index in enumerate(indices):
                bl = self.blmin[tuple(sorted([num[i], num[index]]))]
                d = r[j][0]
                if d < bl:
                    e += self.A * (1 + np.cos(d * np.pi / bl))
                    fj = self.A * np.pi * np.sin(np.pi * d / bl)
                    fj /= (d * bl) * v[j]
                    f[index] += fj
        return e, f
