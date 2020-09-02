import os
from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement
from hotcent.slako import SlaterKosterTable


if not os.path.isdir('iter000'):
    os.mkdir('iter000')

element = 'Si'
xc = 'LDA'
configuration = '[Ne] 3s2 3p2'
valence = ['3s', '3p']
occupations = {'3s': 2, '3p': 2}

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
U = atom.get_hubbard_value('3p', scheme='central', maxstep=1.)
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
rmin, dr, N = 0.4, 0.02, 650
sk = SlaterKosterTable(atom, atom)
sk.run(rmin, dr, N, superposition='density', xc=xc, stride=2)
sk.write('iter000/%s-%s_no_repulsion.skf' % (element, element),
         eigenvalues=eigenvalues, spe=0., hubbardvalues=hubbardvalues,
         occupations=occupations)
