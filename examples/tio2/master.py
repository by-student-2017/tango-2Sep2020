import os
import sys
import traceback
from tango import TANGO
from tango.calculators import DftbPlusCalculator, CP2KCalculator
from ga import run_ga, prepare_ga, comparator


def run(args):
    rundir, maxiter = args

    if not os.path.exists(rundir):
        os.mkdir(rundir)
    os.chdir(rundir)

    sys.stdout = open('out.txt', 'a')
    sys.stderr = open('err.txt', 'w')

    try:
        if not os.path.exists('godb.db'): 
            prepare_ga(splits={(2, 2): 1, (2,): 2, (1,): 1}, N=N)
        run_ga(maxiter, kptdensity=kptdensity)
    except:
        traceback.print_exc()
        sys.stderr.flush()
        raise

    os.chdir('..')
    return


def generator(dbfile='godb.db'):
    prepare_ga(dbfile=dbfile, splits={(2, 2): 1, (2,): 2, (1,): 1}, N=20)
    return


if __name__=='__main__':
    elements = ['Ti', 'O']
    rcuts = None
    powers = {'O-O': range(2, 7), 'O-Ti': range(2, 7), 'Ti-Ti': range(2, 7)}
    kptdensity = 1.5 

    calc = TANGO(elements,
                 DftCalc=CP2KCalculator,
                 DftbPlusCalc=DftbPlusCalculator,
                 kptdensity=kptdensity,
                 initial_training='random_vc_relax',
                 generator=generator,
                 maximum_angular_momenta={'Ti': 2, 'O': 1},
                 rcuts=rcuts,
                 powers=powers,
                 update_rcuts=False,
                 kBT=1.,
                 comparator=comparator,
                 max_select=20,
                 )

    os.environ['ASE_CP2K_COMMAND'] = 'mpirun -np 20 cp2k_shell.popt'

    N = 1
    for i in range(5):
        calc.run(steps=1, go_steps=0, recalculate_dftb=True,
                 run_go=run, number_of_go_runs=20, restart_go=True)

    N = 20
    calc.max_select = 100
    calc.score_limit = 15.
    calc.run(steps=1, go_steps=50, recalculate_dftb=True,
             run_go=run, number_of_go_runs=20, restart_go=True)
    calc.run(steps=1, go_steps=50, recalculate_dftb=True,
             run_go=run, number_of_go_runs=20, restart_go=False)
    calc.run(steps=1, number_of_go_runs=0, recalculate_dftb=True)
