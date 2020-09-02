from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement
from hotcent.slako import SlaterKosterTable


element = 'C'
xc = 'LDA'
configuration = '[He] 2s2 2p2'
valence = ['2s', '2p']
occupations = {'2s': 2, '2p': 2}

# ------------------------------------
# Compute eigenvalues of the free atom
# ------------------------------------

atom = AtomicDFT(element,
                 xc=xc,
                 configuration=configuration,
                 valence=valence,
                 scalarrel=False,
                 )
atom.run()
eigenvalues = {nl: atom.get_eigenvalue(nl) for nl in valence}

# ---------------------------------------
# Compute Hubbard values of the free atom
# ---------------------------------------

atom = AtomicDFT(element,
                 xc=xc,
                 configuration=configuration,
                 valence=valence,
                 scalarrel=False,
                 confinement=PowerConfinement(r0=40., s=4),
                 )
U = atom.get_hubbard_value('2p', scheme='central', maxstep=1.)
hubbardvalues = {'s': U}

# -------------------------------
# Compute Slater-Koster integrals
# -------------------------------

r_cov = covalent_radii[atomic_numbers[element]] / Bohr
r_wfc = 2 * r_cov
r_rho = 3 * r_cov
atom = AtomicDFT(element,
                 xc=xc,
                 confinement=PowerConfinement(r0=r_wfc, s=2),
                 wf_confinement=PowerConfinement(r0=r_rho, s=2),
                 configuration=configuration,
                 valence=valence,
                 scalarrel=False,
                 )
atom.run()

# Compute Slater-Koster integrals:
rmin, dr, N = 0.5, 0.05, 250
sk = SlaterKosterTable(atom, atom)
sk.run(rmin, dr, N, superposition='density', xc=xc)
sk.write('%s-%s_no_repulsion.skf' % (element, element),
         eigenvalues=eigenvalues, spe=0., hubbardvalues=hubbardvalues,
         occupations=occupations)
