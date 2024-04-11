import logging
import numpy as np
from pathlib import Path
from pyscipopt import Model, quicksum

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
        if self.params['dataset'] == 'SALBP-2013':
            # check that the size of the problem has been specified
            if self.params.get('size', None) is None:
                raise ValueError('The problem size must be specified for ' +
                                 'the dataset `SALBP-2013`.')

        if '.alb' not in self.params['instance']:
            self.params['instance'] += '.alb'

        self.read_alb(
            Path(self.problem_folder, self.params['instance'])
        )

    def salbp1(self, **kwargs):
        N = self.N
        t = self.t
        c = self.c
        P = self.P
        # Create the mip solver with the SCIP backend.
        ins = self.params['instance'].replace('.IN2', '').replace('.alb', '')
        model = Model(f"{self.params['dataset']}_{ins}_{c}")
        if not model:
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
                x[i, k] = model.addVar(vtype="B", name="x_%s_%s" % (i, k))

        y = {}
        for k in range(M):
            y[k] = model.addVar(vtype="B", name="y_%s" % k)

        ###############
        # CONSRTAINTS #
        ###############
        # a product must be assigned to a machine
        for i in range(N):
            model.addCons(
                quicksum([x[i, k] for k in np.arange(N)[FS[i]]]) == 1,
                name="task_assignment_%d" % i)

        # cycle time must be respected
        for k in range(M):
            model.addCons(
                quicksum(
                    [t[i] * x[i, k] for i in np.arange(N)[FS[i]] if FS[i, k]]
                ) <= c * y[k],
                name="cycle_time_station_%d" % k
            )

        # precedence constraints
        for i in range(N):
            for j in np.arange(N)[P[i]]:
                model.addCons(
                    quicksum([k * x[j, k] for k in np.arange(N)[FS[j]]]) <=
                    quicksum([k * x[i, k] for k in np.arange(N)[FS[i]]]),
                    name="precedence_%d_%d" % (i, j)
                )

        # SOS1 constraints
        for i in range(N):
            model.addConsSOS1([x[i, k] for k in np.arange(N)[FS[i]]])

        # write objective function
        model.setObjective(quicksum([y[k] for k in range(M)]), "minimize")

        # collect problem data to a dictionary and append them to the model
        model.data = {
            'N': self.N,
            'C': self.c,
            't': self.t,
            'P': self.P,
            'Px': Px,
            'Fx': Fx,
            'tau': tau,
            'E': E,
            'L': L,
            'FS': FS
        }

        return model

    def _write_model(self, **kwargs):
        if int(self.type) == 1:
            return self.salbp1(**kwargs)


if __name__ == '__main__':
    pass
