from ase.calculators.dftb import Dftb
from ase.calculators.calculator import kptdensity2monkhorstpack


class DftbPlusCalculator(Dftb):
    def __init__(self, atoms, kpts=(1,1,1), use_spline=False,
                 maximum_angular_momenta={}, **kwargs):

        if type(kpts) == float or type(kpts) == int:
            mp = kptdensity2monkhorstpack(atoms, kptdensity=kpts, even=False)
            kpts = tuple(mp)

        if use_spline:
            kwargs['Hamiltonian_PolynomialRepulsive'] = 'SetForAll { No }'
        else:
            kwargs['Hamiltonian_PolynomialRepulsive'] = 'SetForAll { Yes }'

        if type(kpts) == float or type(kpts) == int:
            mp = kptdensity2monkhorstpack(atoms, kptdensity=kpts, even=False)
            kpts = tuple(mp)

        kwargs['Hamiltonian_MaxAngularMomentum_'] = ''
        symbols = atoms.get_chemical_symbols()
        unique_symbols = list(set(symbols))
        for s in unique_symbols:
              key = 'Hamiltonian_MaxAngularMomentum_%s' % s
              maxmom = maximum_angular_momenta[s]
              kwargs[key] = 'spd'[maxmom].__repr__()

        kwargs['Hamiltonian_SCC'] = 'Yes'
        kwargs['Hamiltonian_ShellResolvedSCC'] = 'No'
        Dftb.__init__(self, atoms=atoms, kpts=kpts, **kwargs)

    def exit(self):
        pass
