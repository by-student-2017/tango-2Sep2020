try:
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
except ImportError:
    print('Warning: could not load matplotlib')
import os
import json
import numpy as np
import multiprocessing as mp
from ase.db import connect
from ase.units import Bohr, Hartree
from ase.data import atomic_numbers, atomic_masses
from ase.calculators.calculator import kptdensity2monkhorstpack
from tango import repulsion, utilities, run_utils


class TANGO:
    """ Class for running TANGO (tight-binding approximation
    enhanced global optimization) runs.


    * Arguments in the initialization:

    elements: list of all elements present


    * General keyword arguments in the initialization:

    DftCalc: suitable ASE-style DFT calculator class.
             Should accept an Atoms object as argument,
             as well as the following keyword arguments:
             * kpts: Monkhorst-Pack k-point grid 3-tuple
             * run_type: for isolated atoms or dimers,
               run_type='atom' or 'dimer' will be passed,
                 which may be used to modify the input
             The class must also define an exit() method
             which closes the calculator.

    DftbPlusCalc: suitable ASE-style DFTB+ calculator class.
                  Should accept an Atoms object as argument,
                  as well as the following keyword arguments:
                  * kpts: Monkhorst-Pack k-point grid 3-tuple
                  * use_spline: whether to use spline (True) or
                    polynomial (False) based repulsive potentials
                  * maximum_angular_momenta: a dictionary with
                    the maximum angular momenta for each element
                  The class must also define an exit() method
                  which closes the calculator.

    kptdensity: k-point sampling density (in reciprocal Angstrom)
                for the DFT and DFTB calculations. If None,
                only the Gamma-point is included

    initial_training: 'random_relax', 'random_vc_relax' or 'dimers'.
                The first two schemes rely on (partially) relaxed
                randomly generated structures. In the 'dimers' scheme
                (discouraged) the initial repulsive potentials are
                determined solely from the dimer curves

    generator: method which upon calling generates a database with the
               random initial structures. Should accept a 'dbfile'
               keyword argument with the name for the database file

    maximum_angular_momenta: a dictionary with the highest angular
               momenta for each element, e.g. {'Pt':2, 'H':1}


    * Keyword arguments related to the repulsive potential fitting:

    rcuts: None or dictionary of repulsion cutoff radii for each
           (non-redundant) pair of elements. Missing cutoff radii
           will be automatically chosen based on covalent radii
           (see tango.utilities.get_default_rcut).
           If a cutoff is set to None, the repulsive interactions
           for the pair are omitted (set to zero), which is
           recommended if the fitting database does not contain
           these pairs at sufficiently close (i.e. NN) distances.
           This option furthermore allows to 'recycle' repulsive
           potentials which were created separately and which don't
           need to be refitted. To this end, a <pair>_spline.skf
           (and <inverse_pair>_spline.skf in the heteronuclear case)
           needs to be provided (along with the *_no_repulsion.skf
           files), which contain the pre-made 'Spline' blocks.

    mode: 'poly': use polynomials for the repulsive potentials
          'exp_poly' (default): same as 'poly', except that an
                  exponential function is used at very short
                  inter-atomic distances
          'exp_spline': same as 'exp_poly', but with splines instead
                  of polynomials

    powers: None, or dictionary of polynomial powers to include for
            each (sorted, non-redundant) pair of elements.
            In 'exp_spline' mode, only the highest power is considered,
            which determines the order of the spline polynomials (currently
            this value has to be the same for all element pairs).
            For missing entries, the default powers (2 up to 6) are used
            in 'poly' and 'exp_poly' mode, and the default order in
            'exp_spline' mode is 3 (i.e. cubic splines).

    update_rcuts: whether to update the repulsion cutoff radii
                  based on the radial distribution functions

    force_scaling: additional scaling factor (in distance units) to apply
                   to the forces relative to the cohesive energy. For a
                   structure with (energetical) weight exp(-E/kBT),
                   the weight of each force component will be equal to
                   exp(-E/kBT) * force_scaling / Natoms.

    fit_constant: None, 'element', or 'formula' (default).
                  Whether to allow the fitting of a constant energy
                  shift -- either one for each unique empirical formula
                  in the dataset, or one for each element.

    kBT: energy (in eV) to be used assigning Boltzmann weights to
         each structure based on the per-atom cohesive energies.

         Hence, structures with energies up to around Natoms * kBT
         above the lowest energy structure will be given relatively
         high weights (close to 1), whereas structures with energies
         quite higher than Natoms * kBT above that of the lowest energy
         structure will have low weights (close to 0) in the fitting.

         Typical values are e.g. kBT = 0.1 up to 1.0 for crystal
         structures, and lower values (0.01 - 0.1 eV) for metal
         clusters with many low-lying isomers.

    weight_distribution: parameter in between 0 and 1, relevant
         when the fitting structures have different stoichiometries.
         If 0, the Boltzmann weights for each stoichiometry are
         calculated independently, so that the most stable structures
         of each stoichiometry will receive the same (and highest)
         weights. If 1 (the current default), all structures get weighted
         on the "same" cohesive-energy-per-atom scale. Values in between
         0 and 1 are allowed and represent intermediate schemes.

    kBT_dimer: energy to be used in the Boltzmann weighting in
         fitting the exponential part of the repulsion from the dimer
         curves (for the 'exp_poly' and 'exp_spline modes).

    rmins: None or dictionary of 'minimal' radii for each
         (non-redundant) pair of elements, determining the distance
         where the repulsive potential switches from the exponential
         part to the spline- or polynomial based part. If None, these
         distances are determined based on calculations with atomic
         dimers (see run_utils.run_dimers). Otherwise, such dimer curves
         are not used, and the parameters of the exponential repulsive
         part are fitted to match the 0th, 1st, and 2nd derivatives
         of the spline or polynomial at the 'rmins' distance. Note that,
         in this case, it is strongly recommended that (1) these
         distances are chosen sufficiently short so that one is indeed
         in a sufficiently repulsive regime, (2) the training set
         includes certian structures with at least some interatomic
         distances close to the 'rmins' distances (e.g. by using
         the same values in a 'blmin' dictionary used in generating
         the random initial structures).

    referencing: defines what are the DFT(B) 'reference' energies and
         forces, i.e. which energy differences the DFTB repulsive
         potential fitting is supposed to reproduce. These references
         are substracted from the total DFT(B) energies and forces
         prior to fitting. Can be either of the following:

         -> referencing='atomic' (default): the most common case where
            the cohesive energies are the target energy differences
            (typically modulo a certain shift, see 'fit_constant').
            The (relaxed) structures in the supplied databases ought
            to include the key_value_pairs entries 'e_dft_ref'
            (number or array-like) and 'f_dft_ref' (a 3xNatoms-array).
            If these are not present, they will be put to zero,
            meaning no reference value will be substracted. This will
            only yield useful results if 'fit_constant' is not None.

         -> referencing=a list of lists of integers: this allows to
            define the different 'subsystems' present within each
            structure, of which the energies/forces are to be
            substracted from those of the complete configuration.
            Each list of integers represents the indices of the atoms
            in one subsystem. Example: suppose you have already made
            (or found) parametrizations for a molecule and a metal
            surface, and you want to fit the metal-molecule repulsion
            based on a database of substrate+adsorbate configurations.
            The pre-made splines should then be provided as *_spline.skf
            files (see 'rcuts') and 'referencing' can then e.g. be
            set to [<list of substrate atom indices>, <list of
            adsorbate atom indices>].

    * Keyword arguments regarding the extraction of the
    best unique DFTB structures from a set of GO runs
    (for more info see tango.utilities.extract_best_unique):

    comparator: a suitable structure comparator (typically the same
                comparator as in the GO runs)

    max_select: upper bound on the number of DFTB structures to include.
                If None, no such bound is imposed

    num_stddev: number of standard deviations relative to the
                average score of all candidates, which is used to
                pre-select only the more stable structures.
                If None, this criterion is not applied

    score_limit: lower bound for the raw score relative to the highest
                raw score found in the GO runs


    * Keyword arguments related to the short DFT relaxation step:

    opt_precon: whether to use preconditioned optimizers (PreconLBFGS,
                falling back to PreconFIRE in case of failure).
                In some cases regular BFGS (with fallback to FIRE)
                may give better performance, and can be chosen with
                opt_precon=False.

    opt_maxsteps: how many steps to perform during the (typically
                  rather short) DFT relaxation (default: 20).
    """
    def __init__(self, elements, DftCalc=None, DftbPlusCalc=None,
                 kptdensity=None, initial_training='random_relax',
                 generator=None, maximum_angular_momenta={}, rcuts=None,
                 mode='exp_poly', powers=None, update_rcuts=False,
                 force_scaling=1e-3, fit_constant='formula',
                 kBT=1.0, weight_distribution=1., kBT_dimer=50., rmins=None,
                 referencing='atomic', comparator=None, max_select=200,
                 num_stddev=None, score_limit=None, opt_precon=True,
                 opt_maxsteps=20):

        self.elements = elements

        self.DftCalc = DftCalc
        self.DftbPlusCalc = DftbPlusCalc
        self.kptdensity = kptdensity
        self.initial_training = initial_training
        assert self.initial_training in ['random_relax', 'random_vc_relax',
                                         'dimers']
        self.generator = generator

        self.maximum_angular_momenta = maximum_angular_momenta
        assert np.all([e in maximum_angular_momenta for e in self.elements])

        self.rcuts = rcuts if rcuts is not None else {}
        for pair in self.rcuts:
            # Ensure pairs are ordered alphabetically
            pair2 = '-'.join(sorted(pair.split('-')))
            if pair != pair2:
                self.rcuts[pair2] = self.rcuts[pair]
                del self.rcuts[pair]

        self.mode = mode
        assert self.mode in ['poly', 'exp_poly', 'exp_spline']
        self.powers = powers.copy() if powers is not None else {}
        self.update_rcuts = update_rcuts
        self.force_scaling = force_scaling
        self.fit_constant = fit_constant
        assert self.fit_constant in [None, 'formula', 'element']
        self.kBT = kBT
        self.weight_distribution = weight_distribution
        self.kBT_dimer = kBT_dimer
        self.rmins = rmins
        if self.rmins is not None:
            assert self.mode in ['exp_poly', 'exp_spline']
            assert self.initial_training != 'dimers'
            for pair in self.rmins:
                # Ensure pairs are ordered alphabetically
                pair2 = '-'.join(sorted(pair.split('-')))
                if pair != pair2:
                    self.rmins[pair2] = self.rmins[pair]
                    del self.rmins[pair]

        self.referencing = referencing

        self.comparator = comparator
        self.max_select = max_select
        self.num_stddev = num_stddev
        self.score_limit = score_limit

        self.opt_precon = opt_precon
        self.opt_maxsteps = opt_maxsteps

        self.prefixes = utilities.get_skf_prefixes(elements, redundant=False)
        self.all_prefixes = utilities.get_skf_prefixes(elements, redundant=True)

        for p in self.prefixes:
            if p not in self.rcuts:
                self.rcuts[p] = utilities.get_default_rcut(*p.split('-'))
            if p not in self.powers:
                if 'poly' in self.mode:
                    self.powers[p] = range(2, 7)
                else: 
                    self.powers[p] = range(4)

    def run(self, steps=1, recalculate_dftb=True, go_steps=None,
            restart_go=True, number_of_go_runs=1, run_go=None):
        """ Method for performing the TANGO runs.

        steps: number of iterations
        recalculate_dftb: whether to recalculate the (repulsion-less)
              DFTB energies and forces of the database structures
        go_steps: number of GO iterations to do within each iteration.
                  Either a single integer or list of integers (one for
                  each iteration). Default is to use 50, 100, 150, ...
        restart_go: whether to restart the GO runs from scratch or to
                    continue the GO runs from the previous parametrization.
                    If True, the GO runs should employ 'godb.db' database
                    files
        number_of_go_runs: number of DFTB GO runs to run in parallel;
                  For example, serial DFTB runs can be employed and
                  parallel GO runs is then chosen equal to the number
                  of available cores on the compute node  
        run_go: method which, upon calling, performs a GO run. It should
                accept a single tuple as argument, which consists of
                (name of GO run directory, maximum number of GO iterations)
        """

        # Try to find where we left last time
        start = 0
        while os.path.exists('iter%03d' % start):
            start += 1
        start = max(0, start - 1)

        for index, step in enumerate(range(start, start + steps)):

            d = 'iter%03d' % step
            if not os.path.exists(d):
                os.mkdir(d)
            os.chdir(d)

            if step == 0 and not os.path.exists('initial.db'):
                if 'random' in self.initial_training:
                    # Generate initial random structures
                    self.generator.__call__(dbfile='initial.db')
                elif 'dimers' in self.initial_training:
                    pass

            if not os.path.exists('PARAMETRIZED'):
                # Carry out the (re)parametrization
                self.parametrize(step, recalculate_dftb=recalculate_dftb)
                os.system('touch PARAMETRIZED')

            if not os.path.exists('GO_FINISHED'):
                # Carry out the GO runs with DFTB
                if go_steps is None:
                    maxiter = (step + 1)*50  # number of GO iterations
                elif type(go_steps) == list:
                    maxiter = go_steps[index]
                elif type(go_steps) == int:
                    maxiter = go_steps
                else:
                    err = 'go_steps parameter is neither None, list or int!'
                    raise ValueError(err)

                if not restart_go and step > 0:
                    # Copy godb.db files from previous step
                    for i in range(number_of_go_runs):
                        d = 'run%03d' % i
                        if not os.path.exists(d):
                            os.mkdir(d)
                            dbfile = '../iter%03d/%s/godb.db' % (step-1, d)
                            os.system('cp %s %s' % (dbfile, d))

                os.environ['DFTB_PREFIX'] = os.getcwd() + '/'
                po = mp.Pool()
                multiproc_args = []
                for i in range(number_of_go_runs): 
                    multiproc_args.append(['run%03d' % i, maxiter])
                harvest = po.map(run_go, multiproc_args, chunksize=1)
                po.close()
                po.join()

                if number_of_go_runs > 0:
                    os.system('touch GO_FINISHED')

            dbfile = 'best_unique_iter%03d.db' % step
            if not os.path.exists(dbfile) and number_of_go_runs > 0:
                utilities.extract_best_unique(self.comparator,
                                              max_select=self.max_select,
                                              num_stddev=self.num_stddev,
                                              score_limit=self.score_limit,
                                              dbfile=dbfile)
            os.chdir('..')
            os.mkdir('iter%03d' % (step + 1))

        print('TANGO run completed.')
        return

    def parametrize(self, step, recalculate_dftb=True):
        """ (Re)parametrizes the DFTB model.

        step: iteration number (equal to the number of
              previous (re)parametrizations)
        recalculate_dftb: whether to recalculate the (repulsion-less)
              DFTB energies and forces of the database structures
        """
        if step == 0:
            if 'random' in self.initial_training:
                dbfile = 'initial.db'
            elif 'dimers' in self.initial_training:
                dbfile = None

            # Check whether we have all required
            # Slater-Koster files
            for p in self.all_prefixes:
                f = '%s_no_repulsion.skf' % p
                if not os.path.exists(f):
                    msg = 'Could not find %s in iter000 directory. ' % f + \
                          'Please use make sure all the "repulsionless" ' + \
                          'SKF files are copied to this location after ' + \
                          'generating them with TANGO ("build_skf_files"),' + \
                          ' Hotbit, Hotcent, or any other software.'
                    raise IOError(msg)

            # Perform isolated atom calculations
            jsonfile = 'atomic_energies.json'
            if os.path.exists(jsonfile):
                with open(jsonfile, 'r') as f:
                    atomic_energies = json.load(f)
            else:
                atomic_energies = {}

            for element in self.elements:
                key = '%s_DFT' % element
                if key not in atomic_energies:
                    e = run_utils.run_atom(element, self.DftCalc)
                    atomic_energies[key] = e
                key = '%s_DFTB' % element
                if key not in atomic_energies:
                    p = element + '-' + element
                    os.system('cp %s_no_repulsion.skf %s.skf' % (p, p))
                    os.environ['DFTB_PREFIX'] = os.getcwd() + '/'
                    e = run_utils.run_atom(element, self.DftbPlusCalc,
                          maximum_angular_momenta=self.maximum_angular_momenta)
                    os.system('rm %s.skf' % p)
                    atomic_energies[key] = e

            with open(jsonfile, 'w') as f:
                json.dump(atomic_energies, f)

            # Run the gas phase dimer calculations, if needed
            dimersdb = 'dimers.db' if self.rmins is None else None
            if dimersdb is not None and not os.path.exists(dimersdb):
                for p in self.prefixes:
                    if self.rcuts[p] is None:
                        continue
                    run_utils.run_dimers(dimersdb, self.DftCalc,
                                         atomic_energies, *p.split('-'))
        else:
            dbfile = 'best_unique_iter%03d.db' % (step - 1)
            if not os.path.exists(dbfile):
                os.system('cp ../iter%03d/%s .' % (step - 1, dbfile))

            # Copy the repulsionless SKF files, optional spline-containing
            # files, and the atomic energies, from the first iteration
            for p in self.all_prefixes:
                os.system('cp ../iter000/%s_*.skf .' % p)
            with open('../iter000/atomic_energies.json', 'r') as f:
                atomic_energies = json.load(f)

        # Perform the required DFT runs:
        if dbfile is not None:
            relax = step == 0
            vc_relax = 'vc' in self.initial_training and relax
            run_utils.run_calc(dbfile, self.DftCalc,
                               kptdensity=self.kptdensity,
                               relax=relax,
                               vc_relax=vc_relax,
                               precon=self.opt_precon,
                               maxsteps=self.opt_maxsteps,
                               atomic_energies=atomic_energies,
                               referencing=self.referencing)

        # Select databases to include in the (re)parametrization
        dimersdb = '../iter000/dimers.db' if self.rmins is None else None
        dbfiles = []
        if self.initial_training != 'dimers':
            dbfiles.append('../iter000/initial.db')

        for s in range(step):
            dbfile = '../iter%03d/best_unique_iter%03d.db' % (s + 1, s)
            if os.path.exists(dbfile):
                dbfiles.append(dbfile)
            else:
                print('Warning: could not find %s' % dbfile)

        res = self.fit_repulsion(dbfiles, dimersdb=dimersdb,
                                 atomic_energies=atomic_energies,
                                 recalculate_dftb=recalculate_dftb,
                                 run_checks=True)
        return

    def fit_repulsion(self, dbfiles, dimersdb=None, atomic_energies={},
                      empirical_formula=True, recalculate_dftb=False,
                      run_checks=True):
        """ Governs the fitting of the repulsive potentials.
        Returns the total residual obtained for the fit.

        dbfiles: list of databases with structures for the fit
        dimersdb: database containing the atomic dimer data (if needed)
        atomic_energies: reference DFT and DFTB energies of the
                         isolated atoms. Missing '<Element>_DFT' and
                         '<Element>_DFTB' entries will be set to 0.
        empirical_formula: whether to divide the stoichiometries
           by their greatest common divisor. If True, a database with
           e.g. 'Ti4O8' and 'Ti6O12' stoichiometries will have
           only one unique stoichiometry (i.e. 'TiO2').
           This influences the calculation of the Boltzmann weights.
        recalculate_dftb: whether to recalculate the repulsion-less
            DFTB energies and forces of the database structures
        run_checks: whether to recalculate the DFTB energies
              (including repulsion) of the database structures
              for consistency checks
        """
        for element in self.elements:
            for suffix in ['_DFT', '_DFTB']:
                key = element + suffix
                if key not in atomic_energies:
                    atomic_energies[key] = 0.

        # assumes all the *-*_no_repulsion.skf files are present
        for pair in self.all_prefixes:
            os.system('cp %s_no_repulsion.skf %s.skf' % (pair, pair))

        os.environ['DFTB_PREFIX'] = os.getcwd() + '/'
        rep = repulsion.RepulsionFitter(verbose=True)

        if 'exp' in self.mode and self.rmins is None:
            assert dimersdb is not None
            db = connect(dimersdb)
            keys = ['e_dftb_no_rep', 'e_dftb_ref']

            for pair in self.rcuts:
                if self.rcuts[pair] is None:
                    continue

                trajectory = []
                args = []
                harvest = []

                for row in db.select():
                    atoms = row.toatoms()
                    if '-'.join(sorted(atoms.get_chemical_symbols())) != pair:
                        continue
                    if 'key_value_pairs' not in atoms.info:
                        atoms.info['key_value_pairs'] = {}
                    kv_pairs = atoms.info['key_value_pairs']

                    if all([k in kv_pairs for k in keys]):
                        energy, energy_ref = [kv_pairs[k] for k in keys]
                        if recalculate_dftb:
                            for k in keys:
                                del kv_pairs[k]
                        else:
                            ref = {'e_dftb_ref': energy_ref, 'f_dftb_ref': 0.}
                            harvest.append([atoms, energy, 0., ref])

                    if not all([k in kv_pairs for k in keys]):
                        arg = [atoms, self.DftbPlusCalc, 1e-8, False,
                               self.maximum_angular_momenta, atomic_energies,
                               'atomic']
                        args.append(arg)

                print('Starting required DFTB calculations for dimers')
                po = mp.Pool(processes=None)
                harvest.extend(po.map(run_utils.run_dftbplus_singlepoint, args,
                                      chunksize=2))
                po.close()
                po.join()

                for (atoms, energy, forces, references) in harvest:
                    if energy is not None:
                        kv_pairs = atoms.info['key_value_pairs']
                        if 'e_dft_ref' in kv_pairs:
                            e = utilities.restore_array(kv_pairs['e_dft_ref'],
                                                        ncol=1)
                        else:
                            msg = 'Missing e_dft_ref entry in key_value_pairs'
                            assert self.referencing == 'atomic', msg
                            sym = atoms.get_chemical_symbols()
                            e = [atomic_energies['%s_DFT' % s] for s in sym]

                        extra_kv_pairs = references
                        extra_kv_pairs['e_dftb_no_rep'] = energy
                        extra_kv_pairs['f_dftb_no_rep'] = forces
                        extra_kv_pairs['e_dft_ref'] = e
                        atoms.info['key_value_pairs'].update(extra_kv_pairs)
                        trajectory.append(atoms)

                        if not all([k in kv_pairs for k in keys]):
                            del extra_kv_pairs['e_dft_ref']
                            del extra_kv_pairs['f_dftb_ref']
                            del extra_kv_pairs['f_dftb_no_rep']
                            e = utilities.convert_array(kv_pairs['e_dftb_ref'])
                            extra_kv_pairs['e_dftb_ref'] = e
                            db.update(row.id, **extra_kv_pairs)

                rep.fit_exponential_from_dimer(trajectory, kBT=self.kBT_dimer,
                                               plot=True)

        if len(dbfiles) == 0: 
            assert self.initial_training == 'dimers'

            for pair in self.rcuts:
                if self.rcuts[pair] is None:
                    continue

                elements = pair.split('-')
                rc = self.rcuts[pair]/Bohr

                lines = '\nSpline\n'
                lines += '1 %.8f\n' % rc 

                a1 = rep.exp_rep[pair][0] / (Bohr ** -1)
                a2 = rep.exp_rep[pair][1] - np.log(Hartree)
                a3 = -np.exp(-a1 * rc + a2)

                lines += '%.8f %.8f %.8f\n' % (a1, a2, a3)
                lines += '%.8f %.8f 0 0 0 0 0 0\n' % (rc - 1e-7, rc)

                os.system("echo \'%s\' >> %s.skf" % (lines, pair))

                if elements[0] != elements[1]:
                    os.system("echo \'%s\' >> %s-%s.skf" % \
                              (lines, elements[1], elements[0]))
             
            return 0

        for pair in self.all_prefixes:
            os.system("echo >> %s.skf" % pair)
            f = '%s_spline.skf' % pair
            if os.path.exists(f):
                os.system("cat %s >> %s.skf" % (f, pair))
            else:
                lines = repulsion.get_dummy_spline()
                os.system("echo \'%s\' >> %s.skf" % (lines, pair))

        keys = ['e_dftb_no_rep', 'e_dftb_ref',
                'f_dftb_no_rep', 'f_dftb_ref']

        for dbfile in dbfiles:
            db = connect(dbfile)
            args = []
            harvest = []
            for row in db.select(relaxed=1):
                atoms = row.toatoms(add_additional_information=True)
                if 'key_value_pairs' not in atoms.info:
                    atoms.info['key_value_pairs'] = {}
                kv_pairs = atoms.info['key_value_pairs']

                if all([k in kv_pairs for k in keys]):
                    energy, forces, e_ref, f_ref = [kv_pairs[k] for k in keys]
                    forces = utilities.restore_array(forces)
                    f_ref = utilities.restore_array(f_ref)
                    refs = {'e_dftb_ref': e_ref, 'f_dftb_ref': f_ref}
                    if recalculate_dftb:
                        for k in keys:
                            del kv_pairs[k]
                    else:
                        harvest.append([atoms, energy, forces, refs])

                if not all([k in kv_pairs for k in keys]):
                    arg = [atoms, self.DftbPlusCalc, self.kptdensity,
                           True, self.maximum_angular_momenta,
                           atomic_energies, self.referencing]
                    args.append(arg)

            print('Starting required DFTB calculations for %s' % dbfile)
            po = mp.Pool(processes=None)
            harvest.extend(po.map(run_utils.run_dftbplus_singlepoint, args,
                                  chunksize=2))
            po.close()
            po.join()

            not_converged = 0
            for (atoms, energy, forces, references) in harvest:
                if any([x is None for x in [energy, forces, references]]):
                    not_converged += 1
                else:
                    kv_pairs = atoms.info['key_value_pairs']
                    msg = 'Missing %s_dft_ref entry in key_value_pairs'
                    if 'e_dft_ref' in kv_pairs:
                        e = utilities.restore_array(kv_pairs['e_dft_ref'],
                                                    ncol=1)
                    else:
                        assert self.referencing == 'atomic', msg % 'e'
                        sym = atoms.get_chemical_symbols()
                        e = [atomic_energies['%s_DFT' % s] for s in sym]
                    if 'f_dft_ref' in kv_pairs:
                        f = utilities.restore_array(kv_pairs['f_dft_ref'],
                                                    ncol=3)
                    else:
                        assert self.referencing == 'atomic', msg % 'f'
                        f = np.zeros((len(atoms), 3))

                    extra_kv_pairs = references
                    extra_kv_pairs['e_dftb_no_rep'] = energy
                    extra_kv_pairs['f_dftb_no_rep'] = forces
                    extra_kv_pairs['e_dft_ref'] = e
                    extra_kv_pairs['f_dft_ref'] = f
                    atoms.info['key_value_pairs'].update(extra_kv_pairs)
                    rep.append_structure(atoms)

                    if not all([k in kv_pairs for k in keys]):
                        del extra_kv_pairs['e_dft_ref']
                        del extra_kv_pairs['f_dft_ref']
                        for k in ['f_dftb_no_rep', 'e_dftb_ref', 'f_dftb_ref']:
                            f = extra_kv_pairs[k]
                            extra_kv_pairs[k] = utilities.convert_array(f)
                        db.update(row.id, **extra_kv_pairs)

            print(('Database %s: %d structures with DFTB' + \
                   ' convergence problems') % (dbfile, not_converged))

        # make RDF plots and update rcuts if requested
        for p in self.prefixes:
            rc = utilities.estimate_rcut(dbfiles, *p.split('-'))
            if self.update_rcuts and self.rcuts[p] is not None:
                self.rcuts[p] = rc

        rcuts = self.rcuts.copy()
        powers = self.powers.copy()
        results = rep.get_coefficients(rcuts=rcuts, powers=powers,
                                   mode=self.mode, rmins=self.rmins,
                                   atomic_energies=atomic_energies,
                                   kBT=self.kBT, plot=True,
                                   force_scaling=self.force_scaling,
                                   fit_constant=self.fit_constant,
                                   weight_distribution=self.weight_distribution,
                                   empirical_formula=empirical_formula)

        # modify SKF header or Spline section
        if self.mode == 'poly':
            for pair in self.rcuts:
                if self.rcuts[pair] is None:
                    continue

                elements = pair.split('-')
                linenr = 3 if elements[0] == elements[1] else 2

                line = results['skf_txt'][pair]
                os.system("sed -i '%ds/.*/%s/' %s.skf" % (linenr, line, pair))

                if elements[0] != elements[1]:
                    os.system("sed -i '%ds/.*/%s/' %s-%s.skf" % \
                              (linenr, line, elements[1], elements[0]))
        else:
            for pair in self.rcuts:
                if self.rcuts[pair] is None:
                    splfile = '%s_spline.skf' % pair
                    if os.path.exists(splfile):
                        print('Utilizing %s repulsion from' % pair, splfile)
                        with open(splfile, 'r') as f:
                            lines = ''.join(f.readlines())
                    else:
                        lines = repulsion.get_dummy_spline()
                else:
                    lines = results['skf_txt'][pair]

                os.system("cp %s_no_repulsion.skf %s.skf" % (pair, pair))
                os.system("echo >> %s.skf" % pair)
                os.system("echo \'%s\' >> %s.skf" % (lines, pair))

                pair2 = '-'.join(pair.split('-')[::-1])
                if pair != pair2:
                    os.system("cp %s_no_repulsion.skf %s.skf" % (pair2, pair2))
                    os.system("echo >> %s.skf" % pair2)
                    os.system("echo \'%s\' >> %s.skf" % (lines, pair2))

        use_spline = 'exp' in self.mode

        if run_checks and dimersdb is not None:
            print('Checking -- dimers')
            db = connect(dimersdb)
            args = []

            for row in db.select():
                arg = [row.toatoms(), self.DftbPlusCalc, 1e-8, use_spline,
                       self.maximum_angular_momenta, atomic_energies, 'atomic']
                args.append(arg)

            po = mp.Pool(processes=None)
            harvest = po.map(run_utils.run_dftbplus_singlepoint, args,
                             chunksize=2)
            po.close()
            po.join()

            for (atoms, energy, forces, references) in harvest:
                if energy is None:
                    continue
                r = atoms.get_distance(0, 1)
                e_dft = atoms.get_potential_energy()
                try:
                    e_dft_ref = atoms.info['key_value_pairs']['e_dft_ref']
                    e_dft_ref = utilities.restore_array(e_dft_ref, ncol=1)
                except KeyError:
                    e_dft_ref = [atomic_energies['%s_DFT' % s]
                                 for s in atoms.get_chemical_symbols()]
                e_dft -= np.sum(e_dft_ref)
                e_dftb = energy - np.sum(references['e_dftb_ref'])
                print('%s (r=%.3f) e_dft=%.3f e_dftb=%.3f diff=%.3f' % \
                      (pair, r, e_dft, e_dftb, e_dft - e_dftb))

        if run_checks:
            print('Checking -- structures', flush=True)

            f = open('parity_data.txt', 'w')
            f.write('# DBfile   ID   e_dft   e_dftb   diff\n')

            for dbfile in dbfiles:
                db = connect(dbfile)
                args = []

                for row in db.select(relaxed=1):
                    atoms = row.toatoms()
                    atoms.info['id'] = row.id
                    arg = [atoms, self.DftbPlusCalc, self.kptdensity,
                           use_spline, self.maximum_angular_momenta,
                           atomic_energies, self.referencing]
                    args.append(arg)

                po = mp.Pool(processes=None)
                harvest = po.map(run_utils.run_dftbplus_singlepoint, args,
                                 chunksize=2)
                po.close()
                po.join()

                for (atoms, energy, forces, references) in harvest:
                    if energy is None:
                        continue

                    e_dft = atoms.get_potential_energy()
                    try:
                        e_dft_ref = atoms.info['key_value_pairs']['e_dft_ref']
                        e_dft_ref = utilities.restore_array(e_dft_ref, ncol=1)
                    except KeyError:
                        e_dft_ref = [atomic_energies['%s_DFT' % s]
                                     for s in atoms.get_chemical_symbols()]
                    e_dft -= np.sum(e_dft_ref)

                    e_dftb = energy - np.sum(references['e_dftb_ref'])
                    if self.fit_constant == 'formula':
                        form = atoms.get_chemical_formula(
                                                   empirical=empirical_formula)
                        form_full = atoms.get_chemical_formula(empirical=False)
                        u = utilities.get_formula_units(form_full, form)
                        e_dftb += u * results['constants'][form]
                    elif self.fit_constant == 'element':
                        sym = atoms.get_chemical_symbols()
                        for e, c in results['constants'].items():
                            e_dftb += sym.count(e) * c

                    e_diff = e_dft - e_dftb
                    items = (dbfile, atoms.info['id'], e_dft, e_dftb, e_diff)
                    print('%s #%04d: e_dft=%.3f e_dftb=%.3f diff=%.3f' % items)
                    f.write('%s %d %.6f %.6f %.6f\n' % items)

            print('Checking -- done', flush=True)
            f.close()

        return results['residual']
