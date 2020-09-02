try:
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
except ImportError:
    print('Warning: could not load matplotlib')
    pass
import os
import io
import re
import numpy as np
import multiprocessing as mp
from itertools import combinations_with_replacement
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
from ase.data import covalent_radii, atomic_numbers
from ase.db import connect
from ase.io import read
from ase.ga import get_raw_score
from ase.ga.utilities import get_rdf


def split_formula(formula):
    ''' Splits an ASE-style chemical formula into a list, e.g.
    'Al2O3' -> [('Al', 2), ('O', 3)]
    '''
    l = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    return [(x, int(y) if y != '' else 1) for (x, y) in l]


def get_formula_units(full, empirical):
    full_list = split_formula(full)
    empirical_list = split_formula(empirical)
    u = [x[1] * 1. / y[1] for x, y in zip(full_list, empirical_list)]
    assert np.allclose(u, np.round(u[0])), (u, full, empirical)
    return int(u[0])


def convert_array(arr):
    if np.size(arr) == 1:
        assert np.ndim(arr) == 1
        return arr[0]
    else:
        output = io.StringIO()
        np.savetxt(output, np.copy(arr))
        return output.getvalue()


def restore_array(value, ncol=3):
    t = type(value)
    try:
        isUnicode = t == unicode
    except NameError:
        # In Python3, a str is unicode by default
        isUnicode = False

    if t == str or isUnicode:
        arr = np.array(list(map(float, value.split())))
        return arr.reshape((-1, ncol))
    elif t == float or t == int:
        assert ncol == 1
        return [value]
    else:
        raise ValueError('Unknown type: %s' % t)


def get_skf_prefixes(elements, redundant=False):
    ''' Returns the SKF prefixes for each element pair
    which can be created from a list of elements.

    elements: a list of element symbols
    redundant: whether to also include the 'redundant'
               pairs (i.e. also 'B-A' in addition to 'A-B').
    '''
    if redundant:
        return [x + '-' + y for x in elements for y in elements]
    else:
        c = combinations_with_replacement(sorted(elements), 2)
        return [x + '-' + y for (x, y) in c]


def get_default_rcut(element1, element2, factor=1.5):
    ''' Returns a standard cutoff radius for the repulsive
    interaction between the given elements, proportional
    to the sum of covalent radii.

    element1: the symbol of the first element
    element2: the symbol of the second element
    factor: the proportionality constant
    '''
    num1 = atomic_numbers[element1]
    num2 = atomic_numbers[element2]
    cr1 = covalent_radii[num1]
    cr2 = covalent_radii[num2]
    rcut = factor * (cr1 + cr2)
    return rcut


def estimate_rcut(dbfiles, element1, element2):
    ''' Returns an estimated, optimal cutoff radius for the
    repulsive interaction between the given elements.
    It is estimated as the radius for the first minimum
    after the first maximum of the radial distribution
    function (RDF), as employed in e.g.
    http://dx.doi.org/10.1021/acs.jctc.5b00742

    dbfiles: list of database filenames which contain the
             structures for calculating the RDF
    element1: the symbol of the first element
    element2: the symbol of the second element
    '''
    num1 = atomic_numbers[element1]
    num2 = atomic_numbers[element2]
    cr1 = covalent_radii[num1]
    cr2 = covalent_radii[num2]
    rlim = 2 * (cr1 + cr2)
    d = 0.1
    nbins = int(rlim / d)
    rdf_tot = np.zeros(nbins)
    dist = None

    for dbfile in dbfiles:
        db = connect(dbfile)
        for row in db.select(relaxed=1):
            atoms = row.toatoms()
            sym = atoms.get_chemical_symbols()
            if element1 not in sym or element2 not in sym:
                continue
            cell = atoms.get_cell()
            vol = atoms.get_volume()
            rep = [1, 1, 1]
            for i in range(3):
                if atoms.pbc[i]:
                    axb = np.cross(cell[(i + 1) % 3, :], cell[(i + 2) % 3, :])
                    h = vol / np.linalg.norm(axb)
                    rep[i] = int(np.ceil((2.001 * rlim) / h))
            a = atoms.repeat(rep)
            rdf, dist = get_rdf(a, rlim, nbins, elements=(num1, num2))
            rdf_tot += rdf

    if dist is None:
        print('Warning: no %s-%s nearest neighbours in data sets' % \
              (element1, element2))
        return 0.

    rdf_smooth = gaussian_filter1d(rdf_tot, sigma=1, mode='reflect')

    imax = len(dist) - 1
    for imax in find_peaks(rdf_smooth)[0]:
        if rdf_smooth[imax] > 0.5 * np.max(rdf_smooth):
            break

    imin = min(imax + 1, len(dist) - 1)
    for imin in find_peaks(-rdf_smooth)[0]:
        if imin > imax:
            break

    rmin = dist[imin]
    rmax = dist[imax]
    rcut = rmin if imin > imax else rlim
    print('%s-%s RDF: 1st maximum = %.3f' % (element1, element2, rmax))
    print('%s-%s RDF: subsequent minimum = %.3f' % (element1, element2, rmin))
    print('%s-%s RDF: suggested rcut = %.3f' % (element1, element2, rcut))

    try:
        plt.plot(dist, rdf_tot, '--', label='as-is')
        plt.plot(dist, rdf_smooth, '-', label='smoothened')
        plt.plot([rmax, rmax], [0, max(rdf_tot)], label='first maximum')
        plt.plot([rcut, rcut], [0, max(rdf_tot)], label='suggested rcut')
        plt.xlabel('%s-%s distance [Angstrom]' % (element1, element2))
        plt.ylabel('RDF [-]')
        plt.legend(loc='upper left')
        plt.savefig('%s-%s_rdf.pdf' % (element1, element2))
        plt.clf()
    except Exception as err:
        print(err.message)

    return rcut


def compare(args):
    # Wrapper function for use with multiprocessing
    index, comp, a1, a2, return_dist = args
    if return_dist:
       result = comp._compare_structure_(a1, a2)
    else:
       result = comp.looks_like(a1, a2)
    return (index, result)


def get_unique(args):
    ''' Returns the unique ones from a list of structures.
    In case two structures are the same, the one with
    the highest raw score is included.

    args: (comp, trajectory, use_mp) tuple,
          where comp is the comparator,
          trajectory the list of structures,
          and use_mp whether to use multiprocessing.
    '''
    comp, trajectory, use_mp = args
    selection = []

    for counter, atoms in enumerate(trajectory):
        if not selection:
            selection = [atoms]
            continue

        if use_mp and counter % 10 == 0:
            print('Processed %d out of %d' % (counter, len(trajectory)),
                  flush=True)

        if use_mp:
            args = []
            for i, a in enumerate(selection):
                args.append([i, comp, atoms, a, False])
            po = mp.Pool(processes=None)
            harvest = po.map(compare, args, chunksize=10)
            po.close()
            po.join()
            results = {i: lookalike for (i, lookalike) in harvest}

        unique = True
        worse_lookalikes = []

        for i, a in enumerate(selection):
            verdict = results[i] if use_mp else comp.looks_like(atoms, a)
            if verdict:
                unique = False
                if get_raw_score(atoms) > get_raw_score(a):
                    worse_lookalikes.append(i)

        selection = [a for i, a in enumerate(selection)
                     if i not in worse_lookalikes]
        if unique or len(worse_lookalikes) > 0:
            selection.append(atoms)

    return selection


def extract_best_unique(comparator, max_select=None, num_stddev=None,
                        score_limit=None, dbfile='best_unique.db'):
    ''' Writes a database containing the best unique structures
    from a set of global optimization runs in the current
    working directory. These runs must have written an
    'all_candidates.traj' file containing all the structures
    with their raw scores.

    comparator: a class instance with suitable _compare_structure_
                and looks_like methods for comparing two structures
    max_select: upper bound on the number of best unique structures
                to select. If None (default), no bound is enforced.
    num_stddev: number of standard deviations relative to the
                average score of all candidates, which is used to
                pre-select only the more stable structures.
                Setting it to zero means all better-than-average
                structures are considered for further selection.
                Three standard deviations around the mean is used as
                cutoff in determining the average, to exclude very
                low-score outliers.
    score_limit: as an alternative to num_stddev, this argument sets
                 the minimal raw score for structures to be included.
    dbfile: name of the database where the final selection will be saved.
    '''
    db = connect(dbfile)
    all_candidates = []
    all_cand_dict = {}

    for (dirpath, dirnames, filenames) in os.walk('.'):
        if 'all_candidates.traj' in filenames and 'run' in dirpath:
            print('Found run directory', dirpath)
            candidates = read(dirpath + '/all_candidates.traj@:')
            all_candidates.extend(candidates)
            all_cand_dict[dirpath] = candidates

    all_candidates.sort(key=lambda x: get_raw_score(x), reverse=True)
    raw_scores = np.array([get_raw_score(atoms) for atoms in all_candidates])
    std = np.std(raw_scores)
    mean = raw_scores[len(raw_scores) // 2]
    min_score = mean - 3 * std

    izero = np.argmax(raw_scores < min_score)
    if izero != 0:
        raw_scores = raw_scores[:izero]
        all_candidates = all_candidates[:izero]

    average = np.mean(raw_scores)
    std = np.std(raw_scores)
    max_score = np.max(raw_scores)
    min_score = np.min(raw_scores)

    if num_stddev is not None:
        cut_score = average - num_stddev * std
    elif score_limit is not None:
        cut_score = max_score - score_limit
    else:
        cut_score = min_score

    print('Average = %.3f, Std. dev = %.3f' % (average, std))
    print('N = %d before selecting unique structures' % len(all_candidates))
    print('Max score = %.3f, min score = %.3f, cut score = %.3f' % \
          (max_score, min_score, cut_score), flush=True)

    args = []
    for key, val in all_cand_dict.items():
        raw_scores = np.array([get_raw_score(atoms) for atoms in val])
        izero = np.argmax(raw_scores < cut_score)
        if izero != 0:
            val = val[:izero]
        args.append([comparator, val, False])

    po = mp.Pool(processes=None)
    harvest = po.map(get_unique, args, chunksize=1)
    po.close()
    po.join()

    all_candidates = [atoms for allcand in harvest for atoms in allcand]
    all_candidates.sort(key=lambda x: get_raw_score(x), reverse=True)
    raw_scores = [get_raw_score(atoms) for atoms in all_candidates]

    best_unique = []
    print('N_unique = %d before next selection round' % len(all_candidates),
          flush=True)
    best_unique = get_unique([comparator, all_candidates, True])
    best_unique.sort(key=lambda x: get_raw_score(x), reverse=True)

    N = len(best_unique)
    print('N_unique = %d before further refinement' % N, flush=True)

    if max_select is None or max_select >= N:
        selection = range(N)
    else:
        selection = range(max_select)

    print('Selected indices:', selection)
    for i in selection:
        atoms = best_unique[i]
        raw_score = get_raw_score(atoms)
        db.write(atoms, raw_score_from_ga=raw_score, gaid=i, relaxed=0)

    print('N_unique = %d after final refinement' % len(selection))
    return
