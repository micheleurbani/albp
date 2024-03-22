import numpy as np
from itertools import product
from pathlib import Path
from albp.data.problem import Problem
from pyscipopt import Model, quicksum


class PALBP(Problem):
    """
    The Multiproduct Parallel Lines Assembly Line Balancing Problem (PALBP) is
    an extension of SALBP-1 with several parallel manufacturing lines. Each of
    those lines manufactures another product. All lines share the same global
    cycle time. Based on ordering the lines in a favourable manner, the line
    efficiency can be improved by combining station loads of neighbouring
    lines, which can be served by one operator instead of two, in the balancing
    step. That is, the balancing problems of all lines are interconnected with
    each other via the problem of arranging the lines. The objective is to
    minimize the required number of operators.
    [[Source]](https://assembly-line-balancing.de/palbp/)

    Test instances for PALBP as used in the paper of Gökçen al. (2006). Each
    test problem type includes four components: o1,o2,z1,z2. o1 and z1
    represent the precedence relations of the model produced on first parallel
    assembly line and task times of the model respectively. Similarly, o2 end
    z2 represent the precedence relations of the (another) model produced on
    second parallel assembly line and task times of the model respectively.
    Cycle times can be obtained from the article.
    """

    def __init__(self, params: dict, type: str, solver_id: str) -> None:
        super().__init__(params, type, solver_id)

        self.problem_folder = Path(self.problem_folder, self.params['dataset'])
        self.model_folder = Path(self.model_folder, self.params['dataset'])
        self.model_folder.mkdir(parents=True, exist_ok=True)

    def _retrieve_data(self):
        problem_path = Path(self.problem_folder, self.params['instance'])
        H = 2
        self.N = np.zeros(H, dtype=int)
        # read task times from z1.txt and z2.txt
        time = {}
        for i, z in enumerate(['z1', 'z2']):
            with open(Path(problem_path, f'{z}.txt'), 'r') as f:
                time[z] = f.read().replace('"', '').strip().split('\n')
            time[z] = [int(t) for t in time[z]]
            self.N[i] = len(time[z])
        self.t = np.empty(shape=(H, ), dtype=object)
        for i in range(H):
            self.t[i] = np.array(time[f'z{i + 1}'])
        # read precedence relations in o1.txt and o2.txt
        self.P = np.empty(shape=(H, ), dtype=object)
        for i, o in enumerate(['o1', 'o2']):
            self.P[i] = np.zeros((self.N[i], self.N[i]), dtype=bool)
            with open(Path(problem_path, f'{o}.txt'), 'r') as f:
                P = [j.split('\t') for j in
                     f.read().replace('"', '').strip().split('\n')]
                for k, v in P:
                    self.P[i][int(k) - 1, int(v) - 1] = 1

        self.c = np.max([np.max(t) for t in self.t])

    def palbp1(self, active: bool = False, eps: float = 1e-3, **kwargs):

        model = Model(
            f"{self.params['dataset']}_{self.params['instance']}_{self.c}"
        )
        if not model:
            return

        t = self.t
        c = self.c

        # the number of lines is set equal to 2 according to the dataset
        H = len(self.N)
        # the number of products (equal to the number of lines so far)
        P = H
        # the max number of stations is set to the max number of activities
        K = np.max(self.N)
        # a lower bound on the number of stations
        #TODO: check if self.c is correct

        #############
        # VARIABLES #
        #############
        # if task p, j is assigned to workplace h, k
        x = {}
        for p in range(P):
            for j in range(self.N[p]):
                for h in range(H):
                    for k in range(K):
                        x[p, j, h, k] = \
                            model.addVar(vtype="B",
                                         name="x_%s_%s_%s_%s" % (p, j, h, k))

        # product p is assigned to line h
        y = {}
        for p in range(P):
            for h in range(H):
                y[p, h] = model.addVar(vtype="B", name="y_%s_%s" % (p, h))

        # if products p and q are assigned to lines h and h + 1
        w = {}
        for p, q in product(range(P), range(P)):
            if p != q:
                for h in range(H - 1):
                    w[p, q, h] = model.addVar(vtype="B",
                                              name="q_%s_%s_%s" % (p, q, h))

        # if a workplace is installed at station h, k
        z = {}
        for h in range(H):
            for k in range(K):
                z[h, k] = model.addVar(vtype="B", name="z_%s_%s" % (h, k))

        ############
        # OBJCTIVE #
        ############
        if not active:
            model.setObjective(
                quicksum(
                    [quicksum([z[h, k] for h in range(H)])
                     for k in range(K)])
            )
        elif active:
            model.setObjective(
                quicksum([quicksum([z[h, k] for h in range(H)])
                          for k in range(K)]) + \
                eps * (quicksum([quicksum([k * z[h, k] for k in range(K)])
                                 for h in range(H)]))
            )

        ###############
        # CONSRTAINTS #
        ###############
        # PRODUCT-LINE ASSIGNMENT
        # each product p is assigned to exactly one line h
        for p in range(P):
            model.addCons(quicksum([y[p, h] for h in range(H)]) == 1)

        # each line h gets only one product p
        for h in range(H):
            model.addCons(quicksum([y[p, h] for p in range(P)]) == 1)

        # product-line assignments, products p and q are assigned to
        # neighbouring lines h and h + 1
        for p, q in product(range(P), range(P)):
            if p != q:
                for h in range(H - 1):
                    model.addCons(w[p, q, h] <= y[p, h])
                    model.addCons(w[p, q, h] <= y[p, h + 1])
                    model.addCons(w[p, q, h] >= y[p, h] + y[q, h + 1] - 1)

        # TASK-STATION ASSIGNMENT
        # each task j of product p is assigned to one workplace at station k of
        # line h
        for p in range(P):
            for j in range(self.N[p]):
                model.addCons(
                    quicksum([quicksum([x[p, j, h, k] for k in range(K)])
                              for h in range(H)]) == 1
                )

        # CYCLE TIME RESTRICTIONS
        for h in range(H):
            for k in range(K):
                model.addCons(
                    quicksum(
                        [quicksum([t[p][j] * x[p, j, h, k]
                                   for j in range(self.N[p])])
                    for p in range(P)]) <= c * z[h, k]
                )

        # PRECEDENCE RELATIONS
        for p in range(P):
            for i, j in zip(*np.nonzero(self.P[p])):
                model.addCons(
                    quicksum(
                        [quicksum([k * x[p, i, h, k] for k in range(K)])
                    for h in range(H)]) <=
                    quicksum(
                        [quicksum([k * x[p, j, h, k] for k in range(k)])
                    for h in range(H)])
                )

        # RELATING TASKS AND LINES
        # task p, j must be assigned that line h where product p is
        # manufactured or line h - 1
        for p in range(P):
            for j in range(self.N[p]):
                for h in range(1, H):
                    model.addCons(
                        quicksum([x[p, j, h - 1, k] + x[p, j, h, k]
                                  for k in range(K)]) >= y[p, h]
                    )

        # a split workplace linking station k of line h - 1 and h is installed
        # at (h - 1, k) such that no additional workplace is installed at
        # (h, k)
        for p in range(P):
            for j in range(self.N[p]):
                for h in range(1, H):
                    for k in range(K):
                        model.addCons(
                            x[p, j, h - 1, k] + y[p, h] + z[h, k] <= 2
                        )

        # in case of line 1, no downward linkage of stations is possible
        for p in range(P):
            for j in range(self.N[p]):
                model.addCons(
                    quicksum([x[p, j, 0, k] for k in range(K)]) <= y[p, 1]
                )

        # at a normal workplace, the workload of which only contains tasks of a
        # single product q, is installed at the time that is chosen to
        # manufacture q
        for p, q in product(range(P), range(P)):
            if p != q:
                for h in range(H - 1):
                    for k in range(K):
                        for i in range(self.N[q]):
                            model.addCons(
                                x[q, i, h, k] <= \
                                    quicksum(
                                        [x[p, j, h, k]
                                         for j in range(self.N[p])]
                                    ) + (1 - w[p, q, h])
                            )

        # collect problem data in a dictionary
        model.data = {
            'N': self.N,
            't': self.t,
            'H': H,
            'P': self.P,
            'M': K
        }

        return model

    def _write_model(self, **kwargs):
        if int(self.type) == 1:
            return self.palbp1(**kwargs)


if __name__ == '__main__':
    pass
