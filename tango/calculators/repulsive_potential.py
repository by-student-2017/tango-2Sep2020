""" A simple ASE-style calculator of the repulsive energy
and forces (read from SKF files) with supporting methods
and classes.
"""
import os
import numpy as np
from scipy.spatial.distance import cdist
from ase.units import Hartree, Bohr
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator
from tango.utilities import get_skf_prefixes


class Exponential:
    """ Represents an exponential function exp(-a1 * r + a2) + a3. """
    def __init__(self, a1, a2, a3):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def __call__(self, r, der=0):
        assert der in [0, 1]
        if der == 0:
            result = np.exp(-self.a1 * r / Bohr + self.a2) + self.a3
            return result * Hartree
        elif der == 1:
            result = -self.a1 * np.exp(-self.a1 * r / Bohr + self.a2)
            return result * Hartree / Bohr


class Spline:
    """ Represents a spline based on a series of knots, spline
    coefficients, and a cutoff radius rcut.
    """
    def __init__(self, knots, coeffs, rcut):
        self.knots = np.array(knots)
        self.coeffs = np.array(coeffs)
        self.rcut = rcut
        self.powers = np.arange(len(coeffs[0]))

    def __call__(self, r, der=0):
        assert der in [0, 1]
        out = r > self.rcut * Bohr
        result = np.zeros_like(r)
        result[out] = 0.
        if np.all(out):
            return result

        indices = np.where(~out)[0]
        for i, rr in zip(indices, r[~out] / Bohr):
            index = np.argmax(self.knots > rr) - 1
            assert index > -1, (rr, self.knots[0])
            dr = rr - self.knots[index]
            c = self.coeffs[index]
            if der == 0:
                result[i] = np.sum(c * dr ** self.powers)
                result[i] *= Hartree
            elif der == 1:
                result[i] = np.sum(self.powers * c * dr ** (self.powers - 1))
                result[i] *= Hartree / Bohr

        return result


class ExponentialSpline:
    """ Piecewise function consisting of an exponential part Exp
    (for r < rmin) and a spline part Spl (for rmin < r < rcut).
    """
    def __init__(self, Exp, Spl, rmin, rcut):
        self.Exp = Exp
        self.Spl = Spl
        self.rmin = rmin
        self.rcut = rcut

    def __call__(self, r, der=0):
        rmin = self.rmin * Bohr
        rcut = self.rcut * Bohr
        s = np.array([r]) if np.ndim(r) == 0. else np.array(r)
        return np.piecewise(s, [s < rmin, (rmin <= s) & (s <= rcut), rcut < s],
                            [lambda x: self.Exp(x, der=der),
                             lambda x: self.Spl(x, der=der),
                             lambda x: 0.])


def read_spline_from_skf(filename):
    """ Reads a SKF file and returns an ExponentialSpline
    instance with the exponential+spline-based repulsive
    potential.
    """
    with open(filename, 'r') as f:
        while True:
            if 'Spline' in f.readline():
                break
        else:
            raise IOError('SKF %s does not contain Spline section' % f)

        nInt, rcut = list(map(float, f.readline().split()))
        a1, a2, a3 = list(map(float, f.readline().split()))

        knots = []
        coeffs = []
        for i in range(int(nInt)):
            l = list(map(float, f.readline().split()))
            if len(l) == 6:
                l.extend([0., 0.])
            else:
                assert len(l) == 8
            knots.append(l[0])
            coeffs.append(l[2:])

        knots.append(l[1])
        rmin = knots[0]

        exp = Exponential(a1, a2, a3)
        spl = Spline(knots, coeffs, rcut)
        return ExponentialSpline(exp, spl, rmin, rcut)


class RepulsivePotential(Calculator):
    """ ASE Calculator representing the repulsive potentials in a DFTB
    parameter set. Arguments:

    atoms: an ASE atoms object to which the calculator will be attached

    skfdict: dict with the (paths to the) SKF files containing the
             (exponential+spline-based) repulsive potentials, one
             for every (alphabetically sorted) element pair, e.g.:
             {'O-O':'O-O.skf', 'H-O':'H-O.skf', 'H-H':'H-H.skf'}.

             If equal to None, all files are assumed to reside in
             the $DFTB_PREFIX folder and formatted as *-*.skf
             as in the example above.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms, skfdict=None):
        Calculator.__init__(self)

        elements = np.unique(atoms.get_chemical_symbols())
        self.pairs = get_skf_prefixes(elements, redundant=False)
        rcut = 0.
        self.func = {}

        for p in self.pairs:
            if skfdict is None:
                f = os.environ['DFTB_PREFIX'] + '/%s.skf' % p
            else:
                assert p in skfdict, 'No SKF file specified for %s' % p
                f = skfdict[p]

            assert os.path.exists(f), 'SKF file %s does not exist' % f
            self.func[p] = read_spline_from_skf(f)
            rcut = max([rcut, self.func[p].rcut])

        self.nl = NeighborList([rcut * Bohr / 2.] * len(atoms), skin=1.,
                               bothways=False, self_interaction=False)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        N = len(atoms)
        energy, forces = 0, np.zeros((N, 3))
        cell = atoms.get_cell()
        sym = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        self.nl.update(atoms)

        for i in range(N):
            indices, offsets = self.nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])
            v = pos[i] - p

            for j, index in enumerate(indices):
                p = '-'.join(sorted([sym[i], sym[index]]))
                d = r[j][0]
                energy += self.func[p](d, der=0)
                f = self.func[p](d, der=1) * v[j] / d
                forces[index] += f
                forces[i] -= f

        self.results = {'energy': energy, 'forces': forces}
