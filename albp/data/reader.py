import numpy as np
from albp.data.utils import transitive_closure
from pyscipopt import Reader, SCIP_RESULT, quicksum


SECTION_NUM_TASKS =             "<number of tasks>"
SECTION_CYCLE_TIME =            "<cycle time>"
SECTION_NUM_STATIONS =          "<number of stations>"
SECTION_ORDER_STRENGTH =        "<order strength>"
SECTION_TASK_TIMES =            "<task times>"
SECTION_PRECEDENCES =           "<precedence relations>"
SECTION_TIME_INCREMENTS =       "<sequence dependent time increments>"
SECTION_LINKED_TASKS =          "<linked tasks>"
SECTION_TOTAL_STATION_COST =    "<total station cost>"
SECTION_STATION_COST_UNIT =     "<station cost per unit>"
SECTION_TOTAL_TASK_COST =       "<total task cost>"
SECTION_TASK_COST_UNIT =        "<task cost per unit>"
SECTION_MAX_DEGREE =            "<maximum degree of parallelism>"
SECTION_NUM_EQUIPMENTS =        "<number of equipments>"
SECTION_EQUIPMENT_TASK =        "<equipments per task>"
SECTION_NUM_TAKS_ATTRIBUTES =   "<number of task attributes>"
SECTION_TASK_ATTRIBUTES =       "<task attribute values>"
SECTION_ATTRIBUTE_BOUNDS =      "<attribute bounds per station>"
SECTION_INCOMPATIBLE_TASKS =    "<incompatible tasks>"
SECTION_END =                   "<end>"


class Problem:
    sections: list =                []      # sections found in the data file
    N: int =                        0       # number of tasks
    c: int =                        0       # cycle time
    OS: float =                     None    # order strength
    t: np.ndarray =                 None    # task times
    P: np.ndarray =                 None    # precedence relations
    LT: np.ndarray =                None    # linked tasks (matrix form)
    max_parallelis: np.ndarray =    None    # 


class ALBReader(Reader):

    def readerread(self, path):
        # initialize problem data
        problem = Problem()
        # read problem data from file
        with open(path, "r") as f:
            row = f.readline()
            while row:
                if SECTION_NUM_TASKS in row:
                    problem.N = int(f.readline().strip())
                    problem.sections.append(SECTION_NUM_TASKS)
                elif SECTION_CYCLE_TIME in row:
                    if not problem.c:
                        problem.c = int(f.readline().strip())
                    problem.sections.append(SECTION_CYCLE_TIME)
                elif SECTION_NUM_STATIONS in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_NUM_STATIONS)
                elif SECTION_ORDER_STRENGTH in row:
                    problem.OS = float(f.readline().strip().replace(",", "."))
                    problem.sections.append(SECTION_ORDER_STRENGTH)
                elif SECTION_TASK_TIMES in row:
                    problem.t = np.array([int(f.readline().strip().split(" ")[-1])
                                        for _ in range(problem.N)])
                    problem.sections.append(SECTION_TASK_TIMES)
                elif SECTION_PRECEDENCES in row:
                    problem.P = np.zeros((problem.N, problem.N), dtype=bool)
                    for p in f.readlines():
                        if p.strip():
                            p = tuple(p.strip().split(","))
                            problem.P[int(p[0]) - 1, int(p[-1]) - 1] = 1
                        else: break
                    problem.sections.append(SECTION_PRECEDENCES)
                elif SECTION_TIME_INCREMENTS in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_TIME_INCREMENTS)
                elif SECTION_LINKED_TASKS in row:
                    problem.LT = np.zeros((problem.N, problem.N), dtype=bool)
                    for p in f.readlines():
                        if p.strip():
                            p = tuple(p.strip().split(","))
                            problem.LT[int(p[-1]) - 1, int(p[0]) - 1] = 1
                        else: break
                    problem.sections.append(SECTION_LINKED_TASKS)
                elif SECTION_TOTAL_STATION_COST in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_TOTAL_STATION_COST)
                elif SECTION_STATION_COST_UNIT in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_STATION_COST_UNIT)
                elif SECTION_TOTAL_TASK_COST in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_TOTAL_TASK_COST)
                elif SECTION_TASK_COST_UNIT in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_TASK_COST_UNIT)
                # Parallel stations are duplicates of some serial station such
                # that the local cycle time is a multiple of the global cycle
                # time.
                # The maximal number of times a station can be installed in
                # parallel
                # TODO: check section name
                elif SECTION_TASK_COST_UNIT in row:
                    problem.max_parallelism = int(f.readline().strip())
                    problem.sections.append(SECTION_TASK_COST_UNIT)
                elif SECTION_NUM_EQUIPMENTS in row:
                    problem.n_equipments = int(f.readline().strip())
                    problem.sections.append(SECTION_NUM_EQUIPMENTS)
                elif SECTION_EQUIPMENT_TASK in row:
                    problem.equipment = \
                        np.array([int(f.readline().strip().split(" ")[-1])
                                    for _ in range(problem.N)])
                    problem.sections.append(SECTION_EQUIPMENT_TASK)
                elif SECTION_NUM_TAKS_ATTRIBUTES in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_NUM_TAKS_ATTRIBUTES)
                elif SECTION_TASK_ATTRIBUTES in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_TASK_ATTRIBUTES)
                elif SECTION_ATTRIBUTE_BOUNDS in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_ATTRIBUTE_BOUNDS)
                elif SECTION_INCOMPATIBLE_TASKS in row:
                    raise NotImplementedError
                    problem.sections.append(SECTION_INCOMPATIBLE_TASKS)
                elif SECTION_END in row:
                    break
                row = f.readline()

        # check that important data are compliant
        assert problem.N > 0
        assert problem.t is not None
        if problem.c < np.max(problem.t):
            problem.c = np.max(problem.t)

        # problem parameters
        M = problem.N
        # transitive closures predecessors
        Px = transitive_closure(problem.P)
        # transitiive closures successors
        Fx = transitive_closure(problem.P.T)
        # compute earliest and latest stations
        tau = problem.t / problem.c  # relative task time
        # compute earliest
        E = np.zeros(problem.N, dtype=int)
        for i in range(problem.N):
            E[i] = np.ceil(tau[i] + np.sum(tau[Px[i]]))
        # compute latest
        L = np.zeros(problem.N, dtype=int)
        for i in range(problem.N):
            L[i] = M + 1 - np.ceil(tau[i] + np.sum(tau[Fx[i]]))
        # compute feasible stations
        FS = np.zeros((problem.N, problem.N), dtype=bool)
        for i in range(problem.N):
            FS[i, E[i]:L[i] + 1] = 1

        #############
        # VARIABLES #
        #############
        x = {}
        for i in range(problem.N):
            for k in np.arange(problem.N)[FS[i]]:
                x[i, k] = self.model.addVar(vtype="B", name="x_%s_%s" % (i, k))

        y = {}
        for k in range(M):
            y[k] = self.model.addVar(vtype="B", name="y_%s" % k)

        ###############
        # CONSRTAINTS #
        ###############
        # a product must be assigned to a machine
        for i in range(problem.N):
            self.model.addCons(
                quicksum([x[i, k] for k in np.arange(problem.N)[FS[i]]]) == 1,
                name="task_assignment_%d" % i)

        # cycle time must be respected
        for k in range(M):
            self.model.addCons(
                quicksum(
                    [problem.t[i] * x[i, k] for i
                     in np.arange(problem.N)[FS[i]] if FS[i, k]]
                ) <= problem.c * y[k],
                name="cycle_time_station_%d" % k
            )

        # precedence constraints
        for i in range(problem.N):
            for j in np.arange(problem.N)[problem.P[i]]:
                self.model.addCons(
                    quicksum(
                        [k * x[j, k] for k in np.arange(problem.N)[FS[j]]]
                    ) <= quicksum(
                        [k * x[i, k] for k in np.arange(problem.N)[FS[i]]]
                    ),
                    name="precedence_%d_%d" % (i, j)
                )

        # write objective function
        self.model.setObjective(quicksum([y[k] for k in range(M)]), "minimize")

        # append data
        self.model.data = x, y, problem

        return {"result": SCIP_RESULT.SUCCESS}


if __name__ == "__main__":
    pass