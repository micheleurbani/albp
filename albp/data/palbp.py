from pathlib import Path
from albp.data.problem import Problem
from ortools.linear_solver import pywraplp


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
        # read precedence relations in o1.txt and o2.txt
        data = {}
        for o in ['o1', 'o2']:
            with open(Path(problem_path, f'{o}.txt'), 'r') as f:
                data[o] = data.replace('"', '').split('\n')
            data[o] = [i.split('\t') for i in data[o]]
            temp = {}
            for k, v in data[o]:
                if not temp.get(int(k), None):
                    temp[int(k)] = []
                temp[int(k)].append(int(v))
            data[o] = temp
        # read task times from z1.txt and z2.txt
        for z in ['z1', 'z2']:
            with open(Path(problem_path, f'{z}.txt'), 'r') as f:
                data[z] = f.read().replace('"', '').split('\n')
            data[z] = [int(t) for t in data[z]]

        return data

    def palbp1(self, **kwargs):

        solver = pywraplp.Solver.CreateSolver(self.solver_id)
        if not solver:
            return

        t = self.t
        c = self.c

        N1 = len(kwargs[f'z1'])
        N2 = len(kwargs[f'z2'])
        #TODO: refine the initial number of stations
        M = N1
        # the number of lines
        H = 2

        #############
        # VARIABLES #
        #############
        x = {}
        for h in range(1, H + 1):
            for i in range(len(kwargs[f'z{h}'])):
                for k in range(M):
                    x[h, i, k] = solver.IntVar(0, 1, "x_%s_%s_%s" % (h, i, k))

        ###############
        # CONSRTAINTS #
        ###############
        # a task must be assigned to one and only one machine
        for h in range(1, H + 1):
            for i in range(len(kwargs[f'z{h}'])):
                solver.Add(solver.Sum([x[h, i, k] for k in range(M)]) == 1)

        # the work content of any station does not exceed the cycle time
        for k in range(K):
            for h in range(1, H + 1):
                solver.Add(
                    solver.Sum([
                        solver.Sum([t[h, i] * x[h, i, k] for h in range(1, H + 1)]),
                        solver.Sum([t[h + 1, i] * x[h + 1, i, k] for h in range(1, H + 2)])
                    ]) <= c)

    def _write_model(self, **kwargs):
        if self.type == '1':
            return self.palbp1(**kwargs)
