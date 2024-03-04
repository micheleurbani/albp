import logging
import numpy as np
from pathlib import Path
from ortools.linear_solver import pywraplp


from albp.data.problem import Problem
from albp.data.utils import transitive_closure

logger = logging.getLogger(__name__)


class SALBP(Problem):

    def __init__(self, params: dict, problem_folder: str, model_folder: str
                 ) -> None:
        super().__init__(params, problem_folder, model_folder)
        self.problem_folder = Path(self.problem_folder, params['dataset'])
        self.model_folder = Path(self.model_folder, params['dataset'])
        self.model_folder.mkdir(parents=True, exist_ok=True)
        if params['dataset'] == 'SALBP-2013':
            self.problem_folder = Path(self.problem_folder, params['size'])
            self.model_folder = Path(self.model_folder, params['size'])
            self.model_folder.mkdir(parents=True, exist_ok=True)

    def _retrieve_data(self) -> None:
        "There are two available data sets, i.e., SALBP-1993 and SALBP-2013."
        if self.params['dataset'] == 'SALBP-1993':
            if '.IN' not in self.params['instance']:
                self.params['instance'] += '.IN2'
            path = Path(self.problem_folder, self.params['instance'])
            with open(path, 'r') as f:
                self.N = int(f.readline())
                self.t = np.array([int(f.readline().strip())
                                   for _ in range(self.N)])
                # define precedence lists
                self.P = np.zeros((self.N, self.N), dtype=bool)
                for p in f.readlines():
                    if p.strip():
                        p = tuple(p.strip().split(','))
                        if p == ('-1', '-1'):
                            break
                        self.P[int(p[1]) - 1, int(p[0]) - 1] = 1
                    else: break

            # try to retrieve the cycle time from params
            if self.params.get('c', None) is None:
                logger.debug("Setting cycle time to max task time duration.")
                self.c = np.max(self.t)
            else:
                self.c = self.params['c']

        elif self.params['dataset'] == 'SALBP-2013':
            # check that the size of the problem has been specified
            if self.params.get('size', None) is None:
                raise ValueError('The problem size must be specified for ' +
                                 'the dataset `SALBP-2013`.')

            if '.alb' not in self.params['instance']:
                self.params['instance'] += '.alb'
            self.read_alb(
                Path(self.problem_folder, self.params['instance'])
            )
        else:
            raise ValueError('Check dataset name.')

    def salbp1(self, **kwargs):
        N = self.N
        t = self.t
        c = self.c
        P = self.P
        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver(self.solver_id)
        if not solver:
            return
        #TODO: think about the implementation of bounds, the assumption M = N is not efficient
        M = self.N

        #################
        # PREPROCESSING #
        #################
        # transitive closures predecessors
        Px = transitive_closure(P)
        # transitiive closures successors
        Fx = transitive_closure(P.T)
        # compute earliest and latest stations
        tau = self.t / c  # relative task time
        # compute earliest
        E = np.zeros(N, dtype=int)
        for i in range(N):
            E[i] = np.ceil(tau[i] + np.sum(tau[Px[i]]))
        # compute latest
        L = np.zeros(N, dtype=int)
        for i in range(N):
            L[i] = M + 1 - np.ceil(tau[i] + np.sum(tau[Fx[i]]))
        # compute feasible stations
        FS = np.zeros((N, N), dtype=bool)
        for i in range(N):
            FS[i, E[i]:L[i] + 1] = 1

        #############
        # VARIABLES #
        #############
        x = {}
        for i in range(N):
            for k in np.arange(N)[FS[i]]:
                x[i, k] = solver.IntVar(0, 1, "x_%s_%s" % (i, k))

        y = {}
        for k in range(M):
            y[k] = solver.IntVar(0, 1, "y_%s" % k)

        ###############
        # CONSRTAINTS #
        ###############
        # a product must be assigned to a machine
        for i in range(N):
            solver.Add(solver.Sum([x[i, k] for k in np.arange(N)[FS[i]]]) == 1)

        # cycle time must be respected
        for k in range(M):
            solver.Add(
                solver.Sum(
                    [t[i] * x[i, k] for i in np.arange(N)[FS[i]] if FS[i, k]]
                ) <= c * y[k]
            )

        # precedence constraints
        for i in range(N):
            for j in np.arange(N)[P[i]]:
                solver.Add(
                    solver.Sum([k * x[j, k] for k in np.arange(N)[FS[j]]]) <=
                    solver.Sum([k * x[i, k] for k in np.arange(N)[FS[i]]])
                )

        # write objective function
        solver.Minimize(solver.Sum([y[k] for k in range(M)]))

        return solver

    def _write_model(self, **kwargs):
        if int(self.type) == 1:
            return self.salbp1(**kwargs)


if __name__ == '__main__':
    pass
