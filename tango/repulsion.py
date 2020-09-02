import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, PPoly
from ase.units import Bohr, Hartree
from ase.neighborlist import NeighborList
from ase.data import atomic_numbers, atomic_masses
from ase.constraints import FixAtoms
from tango.utilities import get_formula_units, split_formula
try:
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
except ImportError:
    # could not load matplotlib
    pass


def exponential(r, a1, a2, a3):
    ''' Returns the value of exp(-a1*r + a2) + a3.
    r: the radius
    a1, a2, a3: parameters (in appropriate units)
    '''
    return np.exp(-a1*r + a2) + a3


def exponential_dr(r, a1, a2, a3):
    ''' Returns the derivative of the above
    exponential function w.r.t r.
    '''
    return -a1*np.exp(-a1*r + a2)


def exponential_deriv(p, a1, a2, a3):
    ''' Returns the derivative of the above
    exponential function w.r.t x.
    '''
    r = np.linalg.norm(p)
    return -a1*np.exp(-a1*r + a2)*p*1./r


def block(matrices):
    ''' Construct block matrix from list of list of matrices;
    Numpy.block not always available (requires >= 1.13.0)
    '''
    mat = np.vstack([np.hstack(m) for m in matrices])
    return mat


def get_dummy_spline():
    lines = 'Spline\n'
    lines += '%d %.8f\n' % (1, 1.0)
    lines += '%.8f %.8f %.8f\n' % (0.0, 0.0, 0.0)
    lines += '%.8f %.8f 0 0 0 0 0 0\n' % (0.5, 1.0)
    return lines


def build_TinvW_matrix(order, spl_r):
    ''' Builds the 'T^-1.W' matrix which relates the lower-order
    spline coefficients to the highest-order coefficients,
    see e.g. the appendix of doi:10.1021/jp902973m
    '''
    spl_n = len(spl_r) - 1
    Q, R = [], []
    for i in range(spl_n):
        Qi = np.zeros((order, order))
        for j in range(order):
            for k in range(order):
                if k < j:
                    continue
                Qi[j, k] = np.math.factorial(k) 
                Qi[j, k] *= 1. / np.math.factorial(k - j)
                Qi[j, k] *= (spl_r[i+1] - spl_r[i]) ** (k - j)
        Q.append(Qi)

        Ri = np.zeros(order)
        for j in range(order):
            Ri[j] = np.math.factorial(order) 
            Ri[j] *= 1. / np.math.factorial(order - j)
            Ri[j] *= (spl_r[i+1] - spl_r[i]) ** (order - j)
        R.append(Ri)

    S = np.identity(order)
    for j in range(order):
        S[j, j] *= -1. * np.math.factorial(j)

    T = []
    for i in range(spl_n):
        t = []
        for j in range(spl_n):
            if j < i:
                t.append(np.zeros((order, order)))
            elif j == i:
                t.append(Q[i])
            elif j == i + 1:
                t.append(S)
            else:
                t.append(np.zeros((order, order)))
        T.append(t)
    T = block(T)

    W = []
    for i in range(spl_n):
        w = []
        for j in range(spl_n):
            if j == i:
                w.append(R[i][:, None])
            else:
                w.append(np.zeros(order)[:, None])
        W.append(w)
    W = block(W)

    return np.dot(np.linalg.inv(T), W)


class PolynomialCoefficientsBuilder:
    ''' Helps buildings the 'A' matrix for polynomial fitting '''
    def __init__(self, rcuts, powers, N, mode):
        self.rcuts = rcuts
        self.powers = powers
        self.N = N
        self.mode = mode
        self.pairs = sorted(self.rcuts.keys())

        # Starting indices for the different pairs:
        self.start_indices = {self.pairs[0]: 0}
        for i in range(1, len(self.pairs)):
            previous = self.pairs[i - 1]
            index = self.start_indices[previous] + len(self.powers[previous])
            self.start_indices[self.pairs[i]] = index

        # Initialize coefficient matrix:
        ncoeff = self.get_number_of_coefficients()
        self.A = np.zeros((self.N, ncoeff))

    def get_max_rcut(self):
        return max(self.rcuts.values())

    def set_energy_row(self, index, pair, r):
        select = r < self.rcuts[pair]
        rr = self.rcuts[pair] - r[select]
        for l, pwr in enumerate(self.powers[pair]):
            self.A[index, self.start_indices[pair] + l] += np.sum(rr ** pwr)

    def set_force_rows(self, index, pair, r, d):
        select = r < self.rcuts[pair]
        rr = self.rcuts[pair] - r[select]
        for l, pwr in enumerate(self.powers[pair]):
            f = pwr * (rr ** (pwr - 1)) / (r[select])
            c_index = self.start_indices[pair] + l
            for m in range(3):
                self.A[index + m, c_index] += np.sum(f * d[select, m])

    def set_exponential_row(self, index, pair, r0):
        for l, pwr in enumerate(self.powers[pair]):
            rr = self.rcuts[pair] - r0
            self.A[index, self.start_indices[pair] + l] = rr ** pwr

    def get_A_matrix(self):
        return self.A

    def get_number_of_coefficients(self):
        return sum([len(self.powers[pair]) for pair in self.powers])

    def get_coefficients(self, x):
        coeffs = {}
        for pair in self.pairs:
            i = self.start_indices[pair]
            c = x[i: i + len(self.powers[pair])]
            coeffs[pair] = c
        return coeffs


class SplineCoefficientsBuilder:
    ''' Helps buildings the 'A' matrix for spline fitting '''
    def __init__(self, knots, order, N, mode):
        self.knots = knots
        self.order = order
        self.N = N
        self.mode = mode
        self.pairs = sorted(self.knots.keys())

        # Starting indices for each (pair, segment) combo;
        # First come the indices for the first pair for all its segments,
        # then the same for the second pair, etc.
        self.Alo_indices = {}
        self.Ahi_indices = {}
        Alo_index, Ahi_index = 0, 0
        for i, pair in enumerate(self.pairs):
            for j in range(len(self.knots[pair]) - 1):
                self.Alo_indices[(pair, j)] = Alo_index
                Alo_index += self.order
                self.Ahi_indices[(pair, j)] = Ahi_index
                Ahi_index += 1

        self.all_segments = self.get_number_of_coefficients()

        # Initialize coefficient matrices:
        self.Alo = np.zeros((self.N, self.order * self.all_segments))
        self.Ahi = np.zeros((self.N, self.all_segments))

        TinvW = []
        for i in range(len(self.pairs)):
            mat = build_TinvW_matrix(self.order, self.knots[self.pairs[i]])
            segments_i = len(self.knots[self.pairs[i]]) - 1
            row = []
            for j in range(len(self.pairs)):
                segments_j = len(self.knots[self.pairs[j]]) - 1
                shape = (segments_i * self.order, segments_j)
                if i == j:
                    assert np.shape(mat) == shape, (np.shape(mat), shape)
                    row.append(mat)
                else:
                    row.append(np.zeros(shape))
            TinvW.append(row)
        self.TinvW = block(TinvW)

    def get_max_rcut(self):
        return max([max(self.knots[pair]) for pair in self.pairs])

    def set_energy_row(self, index, pair, r):
        for k in range(len(self.knots[pair]) - 1):
            select = self.knots[pair][k] <= r
            select *= r < self.knots[pair][k + 1]
            rr = r[select] - self.knots[pair][k]

            for o in range(self.order):
                c_index = self.Alo_indices[(pair, k)] + o
                self.Alo[index, c_index] += np.sum(rr ** o)

            c_index = self.Ahi_indices[(pair, k)]
            self.Ahi[index, c_index] += np.sum(rr ** self.order)

    def set_force_rows(self, index, pair, r, d):
        for k in range(len(self.knots[pair]) - 1):
            select = self.knots[pair][k] <= r
            select *= r < self.knots[pair][k + 1]
            rr = r[select] - self.knots[pair][k]

            for o in range(self.order):
                f = o * (rr ** (o - 1)) / r[select]
                c_index = self.Alo_indices[(pair, k)] + o
                for m in range(3):
                    self.Alo[index + m, c_index] += np.sum(f * -d[select, m])

            f = self.order * (rr ** (self.order - 1)) / r[select]
            c_index = self.Ahi_indices[(pair, k)]
            for m in range(3):
                self.Ahi[index + m, c_index] += np.sum(f * -d[select, m])

    def set_exponential_row(self, index, pair, r0):
        c_index = self.Alo_indices[(pair, 0)]
        self.Alo[index, c_index] = 1.

    def get_A_matrix(self):
        A = self.Ahi - np.dot(self.Alo, self.TinvW)
        return A

    def get_number_of_coefficients(self):
        return sum([len(self.knots[p]) - 1 for p in self.pairs])

    def get_coefficients(self, xhi):
        xlo = -1. * np.dot(self.TinvW, xhi)
        xlo = np.reshape(xlo, (self.all_segments, self.order))
        x = np.hstack((xlo, xhi[:, None]))
        coeffs = {}
        index = 0
        for i, pair in enumerate(self.pairs):
            rows = len(self.knots[pair]) - 1
            coeffs[pair] = x[index: index + rows, :]
            index += rows
        return coeffs


class RepulsionFitter():
    ''' Class for fitting the repulsive interactions.

    verbose: whether to print output to stdout
    dump_matrices: whether to dump the relevant matrices 
                   obtained from least-squares fitting.
    '''
    def __init__(self, verbose=True, dump_matrices=True):
        self.structures = []
        self.exp_rep = {}
        self.exp_rep_r0 = {}
        self.exp_rep_c0 = {}
        self.verbose = verbose
        self.dump_matrices = dump_matrices

    def append_structure(self, atoms):
        ''' Add a structure to the fitting database.

        atoms: an Atoms object, which should contain the
               target (DFT) potential energy and forces

        The required reference DFT and reference DFTB
        energies and forces, as well as the repulsionless
        DFTB energies and forces, should be provided via
        the atoms.info['key_value_pairs'] dict.
        '''
        self.structures.append(atoms)

    def fit_exponential_from_dimer(self, trajectory, kBT=None, plot=True):
        ''' Performs a Levenberg-Marquardt fit of the
        exponential repulsion parameters (a1, a2, a3)
        based on the given dimer curve.

        trajectory: list of Atoms, containing the structures and
                    energies for a set of atomic separations
                    The 'key_value_pairs' key of the info dict of each
                    Atoms object ought to contain the following keys:
                    'e_dft_ref', 'e_dftb_no_rep', 'e_dftb_no_rep_ref'
        kBT: energy to be used in the Boltzmann weighting
        plot: whether to plot the datapoints and the fitted function
        '''
        # Fit the following function: V(r) = exp(-a1*r + a2) + a3
        radii = [] 
        energies = []

        for i, atoms in enumerate(trajectory):
            kv_pairs = atoms.info['key_value_pairs']
            e = atoms.get_potential_energy() - np.sum(kv_pairs['e_dft_ref'])
            e -= kv_pairs['e_dftb_no_rep'] - np.sum(kv_pairs['e_dftb_ref'])
            energies.append(e)
            radii.append(atoms.get_distance(0, 1))

        # initial guess:
        p0 = [1., np.log(np.max(energies) - np.min(energies)), np.min(energies)]

        # assign less weight (sigma > 1) to highest-energy structures:
        if kBT is None:
            sigma = np.ones_like(radii)
        else:
            e = [atoms.get_potential_energy() for atoms in trajectory]
            sigma = [np.exp((e[i] - np.min(e)) / kBT) for i in range(len(e))]

        popt, pcov = curve_fit(exponential, radii, energies, p0=p0,
                               sigma=sigma, maxfev=100000)
        sym = atoms.get_chemical_symbols()
        key = '-'.join(sorted(sym))
        self.exp_rep[key] = tuple(popt)
        self.exp_rep_r0[key] = np.max(radii)
        a1, a2, a3 = popt
 
        if plot:
            plt.plot(radii, energies, 'bo', lw=2)
            r = np.arange(np.min(radii), np.max(radii), 0.01)
            e = exponential(r, *popt)
            plt.plot(r, e, 'r-', lw=2)
            plt.grid()
            plt.savefig('%s_exponential.pdf' % key)
            plt.clf()

        if self.verbose:
            l = '%s exponential: a1=%.3f, a2=%.3f, a3=%.3f, rmin=%.3f, rmax=%.3f'
            print(l % (key, a1, a2, a3, np.min(radii), np.max(radii)))

        return

    def get_stoich_units_weights(self, kBT, atomic_energies,
                                 weight_distribution, empirical_formula):
        ''' Returns a list of the different stoichiometries present
        in the dataset and an array with the weights for each structure.

        The weights are Boltzmann distributed based on per-atom
        cohesive energies:

            w_i = exp( - (E_coh,i - E_coh,ref,i) / (kB * T))

        For weight_distribution = 0, the reference cohesive energy
        for each stoichiometry is simply the lowest (i.e. most negative)
        per-atom cohesive energy for that stoichiometry.

        For weight_distribution = 1, the reference cohesive energies
        are all equal to the lowest per-atom cohesive energy found
        across the different stoichiometries.

        empirical_formula: whether to divide the stoichiometries
           by their greatest common divisor. If True, a database with
           e.g. 'Ti4O8' and 'Ti6O12' stoichiometries will have
           only one unique stoichiometry (i.e. 'TiO2').
        '''
        assert 0 <= weight_distribution <= 1.

        e_coh = []
        all_formulas = []
        units = []

        for i, atoms in enumerate(self.structures):
            sym = atoms.get_chemical_symbols()
            assert all(['%s_DFT' % s in atomic_energies for s in sym])
            e_atom = sum([atomic_energies['%s_DFT' % s] for s in sym])
            e = (atoms.get_potential_energy() - e_atom) / len(atoms)
            e_coh.append(e)
            form = atoms.get_chemical_formula(empirical=empirical_formula)
            all_formulas.append(form)

            # Determine the number of formula units
            form_full = atoms.get_chemical_formula(empirical=False)
            u = get_formula_units(form_full, form)
            units.append(u)

        e_coh = np.array(e_coh)
        all_formulas = np.array(all_formulas)
        formulas = np.unique(all_formulas)
        e_coh_ref = {form: np.min(e_coh[all_formulas == form])
                     for form in formulas}
        e_min = min(e_coh_ref.values())

        for form in formulas:
            if self.verbose:
                print('Lowest cohesive energy for %s: %.5f' % \
                      (form, e_coh_ref[form]))
            e_coh_ref[form] *= (1. - weight_distribution)
            e_coh_ref[form] += weight_distribution * e_min
            if self.verbose:
                print('Reference cohesive energy for %s: %.5f' % \
                      (form, e_coh_ref[form]))

        references = np.array([e_coh_ref[form] for form in all_formulas])
        weights = np.exp(-1. * (e_coh - references) / kBT)
        assert np.all((0 <= weights) * (weights <= 1.))

        return (list(formulas), units, weights)

    def _solve(self, coefficients_builder, rmins, atomic_energies, kBT,
               force_scaling, fit_constant, weight_distribution,
               regularization=0., weigh_residuals=False,
               empirical_formula=True):
        ''' Performs the actual spline or polynomial fitting

        Returns the fitting coefficients, a dictionary with the (optional)
        constants energy shifts for each stoichiometry, as well as the total
        residual and the separate residuals.

        args: see the 'get_coefficients' function

        regularization: the 'Tikhonov factor' used in the L2 regularization
              to damp the spline coefficients of segments which are 'under-
              determined' (e.g. at relatively short interatomic distances
              which may be absent in the structure database)

        weigh_residuals: whether to return the weighted or unweighted
                         residuals

        empirical_formula: whether to divide the stoichiometries
           by their greatest common divisor. If True, a database with
           e.g. 'Ti4O8' and 'Ti6O12' stoichiometries will have
           only one unique stoichiometry (i.e. 'TiO2').
           This influences the calculation of the Boltzmann weights.
        '''
        mode = coefficients_builder.mode
        pairs = coefficients_builder.pairs
        max_rcut = coefficients_builder.get_max_rcut()

        assert mode in ['poly', 'exp_poly', 'exp_spline']

        stoich, units, weights = self.get_stoich_units_weights(kBT,
                                                         atomic_energies,
                                                         weight_distribution,
                                                         empirical_formula)
        elements = list(set([x[0] for y in stoich for x in split_formula(y)]))

        if self.verbose:
            print('Stoichiometries present in data set:', ' '.join(stoich))
            print('Elements present in data set:', ' '.join(elements))

        N = coefficients_builder.N
        b = np.zeros((N,))
        w = np.zeros((N,))

        if fit_constant is None:
            A_cnst = None
        elif fit_constant == 'formula':
            A_cnst = np.zeros((N, len(stoich)))
        elif fit_constant == 'element':
            A_cnst = np.zeros((N, len(elements)))

        counter = 0
        included = []
        pairs_warn = []
        for i, atoms in enumerate(self.structures):
            kv_pairs = atoms.info['key_value_pairs']
            e_dft = atoms.get_potential_energy()
            f_dft = atoms.get_forces()
            pos = atoms.get_positions(wrap=True)
            atoms.set_positions(pos)
            cell = atoms.get_cell()
            sym = np.array(atoms.get_chemical_symbols(), dtype=str)
            form = atoms.get_chemical_formula(empirical=True)
            nat = len(atoms)

            # Setting b- and w-matrix elements;
            # Repulsion from exponentials will be add to b-matrix later
            w[i+counter] = weights[i]
            b[i+counter] = e_dft
            b[i+counter] -= np.sum(kv_pairs['e_dft_ref'])
            b[i+counter] -= kv_pairs['e_dftb_no_rep']
            b[i+counter] += np.sum(kv_pairs['e_dftb_ref'])

            if fit_constant == 'formula':
                index = stoich.index(form)
                A_cnst[i+counter, index] = units[i]
            elif fit_constant == 'element':
                for index, element in enumerate(elements):
                    A_cnst[i+counter, index] = np.count_nonzero(sym == element)

            fixed_indices = []
            for c in atoms.constraints:
                if isinstance(c, FixAtoms):
                    fixed_indices.extend(c.index)

            for j in range(nat):
                if j in fixed_indices:
                    continue

                for k in range(3):
                    df = f_dft[j, k] - kv_pairs['f_dft_ref'][j, k]
                    df -= kv_pairs['f_dftb_no_rep'][j, k]
                    df += kv_pairs['f_dftb_ref'][j, k]
                    index = i + counter + 1 + 3 * j + k
                    b[index] = df
                    w[index] = force_scaling * w[i+counter] / nat

            # Now, the A-matrix elements (energies) 
            nl = NeighborList([max_rcut / 2.] * nat, skin=0., bothways=False,
                              self_interaction=False)
            nl.update(atoms)

            is_valid = True
            for j in range(nat):
                indices, offsets = nl.get_neighbors(j)
                p = pos[indices] + np.dot(offsets, cell)
                r = cdist(p, [pos[j]])

                for k, pair in enumerate(pairs):
                    element1, element2 = pair.split('-')
                    if element1 != sym[j] and element2 != sym[j]:
                        continue
                    other = element2 if sym[j] == element1 else element1
                    select = np.array([sym[indices] == other]).T

                    if 'exp' in mode:
                        if rmins is None:
                            select2 = self.exp_rep_r0[pair] < r
                        else:
                            select2 = rmins[pair] < r

                        if np.sum(select * ~select2) > 0:
                            if self.verbose:
                                print('Warning: exponential %s contribution(s)'
                                      ' involving atom %d in structure %d' % \
                                      (pair, j, i))
                            if rmins is None:
                                exp_e = exponential(r[select * ~select2],
                                                    *self.exp_rep[pair])
                                b[i+counter] -= np.sum(exp_e)
                                select *= select2
                            else:
                                is_valid = False
                                pairs_warn.append(pair)
                                w[i+counter: i+counter+1+3*nat] = 0.

                    coefficients_builder.set_energy_row(i+counter, pair,
                                                        r[select])

            if is_valid:
                included.append(i)
            elif self.verbose:
                print('Omitting structure %d from fit' % i)

            # Next, the forces
            nl = NeighborList([max_rcut / 2.] * nat, skin=0., bothways=True,
                              self_interaction=False)
            nl.update(atoms)

            for j in range(nat):
                if j in fixed_indices or not is_valid:
                    continue

                indices, offsets = nl.get_neighbors(j)
                p = pos[indices] + np.dot(offsets, cell)
                r = cdist(p, [pos[j]])

                for k, pair in enumerate(pairs):
                    element1, element2 = pair.split('-')
                    if element1 != sym[j] and element2 != sym[j]:
                        continue
                    other = element2 if sym[j] == element1 else element1
                    select = np.array([sym[indices] == other]).T

                    if 'exp' in mode and rmins is None:
                        select2 = self.exp_rep_r0[pair] < r
                        d = pos[j] - p[(select * ~select2)[:, 0]]
                        exp_f = exponential_deriv(d, *self.exp_rep[pair])
                        for m in range(3):
                            index = i + counter + 1 + 3 * j + m
                            b[index] -= np.sum(exp_f[:, m])
                        select *= select2

                    d = pos[j] - p[(select)[:, 0]]
                    coefficients_builder.set_force_rows(i+counter+1+3*j, pair,
                                                        r[select], d)
            counter += 3 * nat

        if 'exp' in mode and rmins is None:
            # Add condition that the poly/spline matches the exponential at r0
            for k, pair in enumerate(pairs):
                r0 = self.exp_rep_r0[pair]
                index = N - k - 1
                coefficients_builder.set_exponential_row(index, pair, r0)
                b[index] = exponential(r0, *self.exp_rep[pair])
                w[index] = 1e6

        A = coefficients_builder.get_A_matrix()
        if fit_constant is not None:
            A = np.hstack((A, A_cnst))

        # L2 regularization
        ncoeff = coefficients_builder.get_number_of_coefficients()
        A_loss = np.sqrt(regularization) * np.identity(ncoeff)
        if fit_constant == 'formula':
            A_loss = np.hstack((A_loss, np.zeros((ncoeff, len(stoich)))))
        elif fit_constant == 'element':
            A_loss = np.hstack((A_loss, np.zeros((ncoeff, len(elements)))))
        A = np.vstack((A, A_loss))
        b_loss = np.zeros(ncoeff)
        b = np.hstack((b, b_loss))
        w_loss = np.ones(ncoeff)
        w = np.hstack((w, w_loss))

        print('Structures included in fit: %d out of %d' % \
              (len(included), len(self.structures)))
        if len(included) < len(self.structures):
            assert 'exp' in mode and rmins is not None, (mode, rmins)
            print('Certain structures have been excluded based on:')
            print(' and/or '.join(['rmins[%s]' % p for p in set(pairs_warn)]))
            print('Please check that these parameters have sensible values.')

        x, residual, rank, s = np.linalg.lstsq(A * np.sqrt(w)[:, None],
                                               b * np.sqrt(w), rcond=None)
        if weigh_residuals:
            residuals = b * np.sqrt(w) - np.dot(A * np.sqrt(w)[:, None], x)
        else:
            residuals = b - np.dot(A, x)

        if fit_constant == 'formula':
            constants = {s: c for s, c in zip(stoich, x[-len(stoich):])}
            x = x[:-len(stoich)]
        elif fit_constant == 'element':
            constants = {e: c for e, c in zip(elements, x[-len(elements):])}
            x = x[:-len(elements)]
        else:
            constants = {}

        names = ['A', 'b', 'w', 'x', 'residuals']
        for (m, M) in zip(names, [A, b, w, x, residuals]):
            if self.verbose:
                print('%s matrix:' % m)
                print(M)
            if self.dump_matrices:
                np.save('%s.npy' % m, M)

        coeffs = coefficients_builder.get_coefficients(x)
        return (coeffs, constants, residual, residuals)

    def get_coefficients(self, rcuts=None, powers=None, spl_dr=0.1, rmins=None,
                         mode='exp_spline', atomic_energies={}, kBT=0.1,
                         force_scaling=0.1, plot=True, fit_constant='formula',
                         weight_distribution=1., empirical_formula=True):
        ''' Performs a least-squares fit of polynomial or spline coefficients
        based on the previously registered fitting structure database.

        Returns a dictionary containing the polynomial or spline coefficients,
        the (optional) constants energy shifts for each stoichiometry,
        as well as the residuals and the total residual. Also an 'skf_txt'
        entry is included, which corresponds to the line(s) to be used
        in SKF files. Note that in 'exp_poly' and 'exp_spline' modes,
        the fitted polynomial and spline based potentials are subsequently
        mapped onto (high-resolution) cubic splines for compatibility with
        the SKF format (including the initial exponentially decaying
        repulsion).

        rcuts: dictionary with the cutoff radii for each
               (non-redundant) element pair to be included in the fit.
               If a cutoff is None, the corresponding pair will be
               omitted from the fit (i.e. zero repulsive interaction)
        powers: dictionary with the range of powers for each
               (non-redundant) element pair to be included in the fit.
        spl_dr: desired length of the spline segments (in 'exp_spline'
                mode) in Angstrom.
        rmins: dictionary with distances below which the potentials
               switch to the exponential form (see main.py)
        mode: 'poly', 'exp_poly' or 'exp_spline': the functional form
               of the repulsive potentials.
        atomic_energies: reference DFT and DFTB energies of the
                         isolated atoms. Mandatory.
        kBT: energy to be used in the Boltzmann weighting of the per-atom
             cohesive energies.
        force_scaling: scaling factor (in distance units) to
                       apply to the forces.
        fit_constant: about inclusion of constant energy shifts
                      (see main.py)
        nInt: number of knots for the cubic splines
        plot: whether to plot the resulting repulsive potentials
        weight_distribution: parameter in between 0 and 1, relevant
            when the fitting structures have different stoichiometries.
            If 0, the Boltzmann weights for each stoichiometry are
            calculated independently, so that the most stable structures
            of each stoichiometry will receive the same (and highest)
            weights. If 1, all structures get weighted on the "same"
            cohesive-energy-per-atom scale. Values in between 0 and 1
            are allowed and represent intermediate schemes.
        empirical_formula: whether to divide the stoichiometries
           by their greatest common divisor. If True, a database with
           e.g. 'Ti4O8' and 'Ti6O12' stoichiometries will have
           only one unique stoichiometry (i.e. 'TiO2').
           This influences the calculation of the Boltzmann weights.
        '''
        assert mode in ['exp_spline', 'exp_poly', 'poly']

        for pair in list(rcuts.keys()):
            if rcuts[pair] is None:
                del rcuts[pair]
                continue
            if 'exp' in mode:
                assert pair in self.exp_rep or pair in rmins, \
                       'Missing information on pair: %s' % pair

        pairs = sorted(rcuts.keys())
        assert all([pair in powers for pair in pairs])

        N = len(self.structures)
        N += 3 * sum([len(atoms) for atoms in self.structures])
        N += len(pairs) if 'exp' in mode and rmins is None else 0

        results = {}

        if 'poly' in mode:
            cb = PolynomialCoefficientsBuilder(rcuts, powers, N, mode)
            regularization = 1e-10

        elif 'spline' in mode:
            orders = np.unique([max(powers[pair]) for pair in pairs])
            assert len(orders) == 1
            order = orders[0]

            knots = {} 
            for pair in pairs:
                rmin = self.exp_rep_r0[pair] if rmins is None else rmins[pair]
                knots[pair] = [rmin]
                num = int(np.floor((rcuts[pair] - rmin) / spl_dr))
                knots[pair] += list(np.linspace(rmin, rcuts[pair], num=num,
                                                endpoint=True))
                if self.verbose:
                    items = (pair, order, ' '.join(map(str, knots[pair])))
                    print('Pair %s: order = %d   knots = %s' % items)

            results['order'] = order
            results['knots'] = knots
            cb = SplineCoefficientsBuilder(knots, order, N, mode)
            regularization = 1e-6

        # Fit the polynomial or spline coefficients
        args = (cb, rmins, atomic_energies, kBT, force_scaling, fit_constant,
                weight_distribution)
        kwargs = {'regularization': regularization,
                  'empirical_formula': empirical_formula}
        coeffs, constants, residual, residuals = self._solve(*args, **kwargs)
        results['coeffs'] = coeffs
        results['constants'] = constants
        results['residual'] = residual

        # Construct the text that should either become the 2nd line
        # in the SKF file (in 'poly' mode) or the Spline section (in
        # 'exp_poly' or 'exp_spline' mode)
        results['skf_txt'] = {}

        if mode == 'poly':
            for pair, coeffs in results['coeffs'].items():
                elements = pair.split('-')
                if elements[0] == elements[1]:
                    line = '%.4f ' % atomic_masses[atomic_numbers[elements[0]]]
                else:
                    line = '0.0000 '

                for p in range(2, 10):
                    if p in powers[pair]:
                        index = powers[pair].index(p)
                        c = coeffs[index] 
                        c *= (Bohr ** p) / Hartree
                        line += ' %.8f' % c
                    else:
                        line += ' 0.0'

                line += ' %.8f 10*0.0' % (rcuts[pair] / Bohr)
                results['skf_txt'][pair] = line

        else:
            results['PPoly'] = {}
            if rmins is not None:
                self.exp_rep_r0 = {k: v for k, v in rmins.items()}
                self.exp_rep = {}

            for pair, coeffs in results['coeffs'].items():
                r0 = self.exp_rep_r0[pair]
                rc = rcuts[pair]

                x = np.linspace(r0, rc, endpoint=True, num=1000)
                y = np.zeros_like(x)

                if mode == 'exp_poly':
                    for i, c in enumerate(coeffs):
                        p = powers[pair][i]
                        y += c * ((rc - x) ** p)

                elif mode == 'exp_spline':
                    knots = results['knots'][pair]
                    for k in range(len(knots) - 1):
                        select = knots[k] <= x
                        select *= knots[k + 1] > x
                        rr = x[select] - knots[k]
                        for i, c in enumerate(coeffs[k]):
                            y[select] += c * (rr ** i)

                num = 100
                degree = 3  # SKF format only allows cubic splines
                knots = np.linspace(r0 + 0.05, rcuts[pair] - 0.05,
                                    endpoint=True, num=num)
                tck = splrep(x, y, t=knots, k=degree)
                pp = PPoly.from_spline(tck)
                knots = pp.x[degree:-degree]
                coeffs = pp.c[:, degree:-degree][::-1]

                results['PPoly'][pair] = pp
                elements = pair.split('-')

                if rmins is not None:
                    # fit the exponential parameters to the spline or poly
                    derivs = [pp(r0, nu=nu) for nu in range(3)]
                    if derivs[1] > 0:
                        msg = 'Pair %s -- warning: 1st derivative is ' % pair
                        msg += 'postive at r0=%.3f; setting it to -100' % r0
                        print(msg)
                        derivs[1] = -100.

                    if derivs[2] < 0:
                        msg = 'Pair %s -- warning: 2nd derivative is ' % pair
                        msg += 'negative at r0=%.3f; setting it to 25' % r0
                        print(msg)
                        derivs[2] = 25.

                    a1 = derivs[2] * -1. / derivs[1]
                    xi = derivs[1] * -1. / a1
                    a2 = np.log(xi) + a1 * r0
                    a3 = derivs[0] - xi
                    self.exp_rep[pair] = [a1, a2, a3]

                lines = '\nSpline\n'
                lines += '%d %.8f\n' % (num + 1, rcuts[pair] / Bohr) 

                a1 = self.exp_rep[pair][0] / (Bohr ** -1)
                a2 = self.exp_rep[pair][1] - np.log(Hartree)
                a3 = self.exp_rep[pair][2] / Hartree
                lines += '%.8f %.8f %.8f\n' % (a1, a2, a3)

                line = ''
                order = np.shape(coeffs)[0]
                conversion = np.array([(Bohr ** i) / Hartree
                                       for i in range(order)])
                for i in range(len(knots)-1):
                    coeff = coeffs[:, i] * conversion
                    line += '%.8f %.8f ' % (knots[i] / Bohr, knots[i+1] / Bohr)
                    line += ' '.join(['%.8f' % c for c in coeff])
                    if i == len(knots)-2:
                        dr = (knots[i+1] - knots[i]) / Bohr
                        y0 = sum([c * (dr ** j) for j, c in enumerate(coeff)])
                        y1 = sum([j * c * (dr ** (j - 1))
                                  for j, c in enumerate(coeff)])
                        c4 = (y1 * dr - 5 * y0) / (dr ** 4)
                        c5 = (-y1 * dr + 4 * y0) / (dr ** 5)
                        line += ' %.8f %.8f' % (c4, c5)
                    line += '\n'

                lines += line
                results['skf_txt'][pair] = lines

        # Print out results summary
        if self.verbose:
            print('Residual sum of squares:', results['residual'])
            for k, v in results['constants'].items():
                print('%s %s : fitted constant term = %.6f' % \
                      (fit_constant.title(), k, v))

            for pair, coeff in results['coeffs'].items():
                print('Pair:', pair)
                print('Fitted coefficients:')
                print(coeff)

                if 'poly' in mode:
                    allcoeff = np.zeros(max(powers[pair]) + 1)
                    for i,pwr in enumerate(powers[pair]):
                        allcoeff[pwr] = coeff[i]

                    rc = rcuts[pair]
                    roots = rc - np.roots(allcoeff[::-1])
                    if 'exp' in mode:
                        r0str = ', r0=%.4f' % self.exp_rep_r0[pair]
                    else:
                        r0str = ''

                    print('Roots (cutoff=%.4f%s):' % (rc, r0str))
                    print(roots)

        # Plot the repulsive potentials
        if plot:
            gray = '0.3'
            for pair, coeff in results['coeffs'].items():
                r0 = self.exp_rep_r0[pair] if 'exp' in mode else 0.5
                rc = rcuts[pair]
                output = '%s.pdf' % pair
                startx = r0 - 0.05
                x = np.arange(startx, rc, 0.01)
                y = np.zeros_like(x)

                if 'poly' in mode:
                    label = 'Fitted polynomial'
                    for i, c in enumerate(coeff):
                        pwr = powers[pair][i]
                        y += c * ((rc - x) ** pwr)
                else:
                    label = 'Fitted spline'
                    y = results['PPoly'][pair](x)

                plt.plot(x, y, 'r-', lw=2, label=label)

                xlim = [r0 - 0.1, rc + 0.1]
                ymin = -2. if np.min(y) < 0 else 0
                ymax = np.max(y)
                if 'exp' in mode:
                    er0 = exponential(r0, *self.exp_rep[pair])
                    ymax = max(ymax, er0)
                ymax *= 1.05

                plt.plot([rc, rc], [ymin, ymax], '--', color=gray, lw=2)
                plt.plot(xlim, [0, 0], '--', color=gray, lw=2)
                plt.xlim(xlim)
                plt.ylim([ymin, ymax])

                if 'exp' in mode:
                    if mode == 'exp_poly':
                        y = results['PPoly'][pair](x)
                        label = 'Polynomial mapped on cubic spline'
                        plt.plot(x, y, 'b--', lw=2, label=label)
                    elif mode == 'exp_spline':
                        x = results['knots'][pair]
                        y = list(results['coeffs'][pair][:, 0]) + [0.]
                        label = 'Spline knots'
                        plt.plot(x, y, 'x', color=gray, mew=2, label=label)

                    x = np.arange(startx, r0 + 0.1, 0.01)
                    y = exponential(x, *self.exp_rep[pair])
                    plt.plot(x, y, 'k-', lw=2, label='Exponential')
                    plt.plot([r0, r0], [ymin, ymax], '--', color=gray, lw=2)

                plt.grid()
                plt.xlabel('r (Angstrom)')
                plt.ylabel('Vrep (eV)')
                plt.legend(loc='upper right')
                plt.savefig(output)
                plt.clf()

        return results
