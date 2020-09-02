from ase.units import Bohr
from ase.data import covalent_radii, atomic_numbers
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement
from hotcent.slako import SlaterKosterTable
from tango.repulsion import get_dummy_spline


elements = ['C', 'N']
xc = 'LDA'
configurations = {'C': '[He] 2s2 2p2', 'N': '[He] 2s2 2p3'}
valences = {'C': ['2s', '2p'], 'N': ['2s', '2p']}
occupations = {'C':{'2s': 2, '2p': 2}, 'N': {'2s': 2, '2p': 3}}
eigenvalues, hubbardvalues, atoms = {}, {}, {}


# ------------------------------------------------
# Write dummy *-*_spline.skf files for C-C and N-N
# ------------------------------------------------

for el in elements:
    lines = get_dummy_spline()
    with open('%s-%s_spline.skf' % (el, el), 'w') as f:
        f.write(lines)

# --------------------------------------------------------
# Compute eigenvalues and Hubbard values of the free atoms
# --------------------------------------------------------

for el in elements:
    valence = valences[el]
    atom = AtomicDFT(el,
                     xc=xc,
                     configuration=configurations[el],
                     valence=valence,
                     scalarrel=True,
                     confinement=PowerConfinement(r0=40., s=4),
                     txt='-',
                     )
    u = atom.get_hubbard_value('2p', scheme='central')
    hubbardvalues[el] = {'s': u}
    atom.run()
    eigenvalues[el] = {nl: atom.get_eigenvalue(nl) for nl in valence}

# --------------------------------------------------
# Get KS all-electron ground state of confined atoms
# --------------------------------------------------

atoms = {}
for el in elements:
    r_cov = covalent_radii[atomic_numbers[el]] / Bohr
    r_wfc = 2 * r_cov
    r_rho = 3 * r_cov
    atom = AtomicDFT(el,
                     xc=xc,
                     configuration=configurations[el],
                     valence=valence,
                     scalarrel=False,
                     confinement=PowerConfinement(r0=r_rho, s=2),
                     wf_confinement=PowerConfinement(r0=r_wfc, s=2),
                     txt='-',
                     )
    atom.run()
    atoms[el] = atom

# -------------------------------
# Compute Slater-Koster integrals
# -------------------------------

for i in range(len(elements)):
    el_a = elements[i]
    for j in range(i + 1):
        el_b = elements[j]
        rmin, dr, N = 0.4, 0.02, 500
        sk = SlaterKosterTable(atoms[el_a], atoms[el_b], txt='-')
        sk.run(rmin, dr, N, superposition='density', xc=xc, stride=4)
        if el_a == el_b:
            sk.write('%s-%s_no_repulsion.skf' % (el_a, el_a), spe=0.,
                     eigenvalues=eigenvalues[el_a],
                     hubbardvalues=hubbardvalues[el_a],
                     occupations=occupations[el_a])
        else:
            for pair in [(el_a, el_b), (el_b, el_a)]:
                sk.write('%s-%s_no_repulsion.skf' % pair, pair=pair)
