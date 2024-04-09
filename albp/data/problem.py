import abc
import numpy as np
from pathlib import Path
import pyscipopt


class Problem:

    def __init__(self, params: dict, problem_folder: str,
                 model_folder: str) -> None:
        """
        Parameters
        ----------
        type: str
            One of the values `{1, 2, E, F}`, where `1` means minimise the
            number of stations given the cycle time $C$, `2` means minimise
            the cycle time $C$ with a given number of workstation, `E` means
            maximise line efficiency $E = \frac{\sum_{j=1}^n t_j}{mc}$, and
            `F` indicates the feasibility problem, i.e., whether a feasible
            balance exisits for given vaues of $C$ and $M$.
        solver_id: str
            A string that indicates which solver to call. Available solvers are
            `{SCIP, GUROBI}`.
        params: dict
            A dictionary containing problem-specific parameters. For example:
                - `instance`: the name of the selected instance.
        """
        # problem type
        self.type = params['type']
        # problem parameters
        self.N = None
        if params['c'] is not None:
            self.c = int(params['c'])
        self.t = None
        self.P = None  # adjacency matrix for precedence graph
        self.LT = None  # linked tasks
        self.max_parallelism = None  # the maximum degree of parallelism
        self.n_equipment = None  # number of equipments
        self.equipement = None  # equipments per task

        self.params = params

        if params.get('dataset', None) is None:
            raise ValueError('The dataset name is missing and it must be ' +
                             'specified for the SALBP problem.')
        if params.get('instance', None) is None:
            raise ValueError('Missing instance name.')

        # Problem-specific folders
        self.problem_folder = Path(problem_folder)
        self.model_folder = Path(model_folder)
        self.model_folder.mkdir(parents=True, exist_ok=True)

    def read_alb(self, path: Path) -> None:
        """
        This method reads the "standard" ALB problem description format
        proposed for the data sets found in
        [assembly-line-balancing.de](https://assembly-line-balancing.de/).
        """
        with open(path, "r") as f:
            row = f.readline()
            while row:
                if "<number of tasks>" in row:
                    self.N = int(f.readline().strip())
                elif "<cycle time>" in row:
                    self.c = int(f.readline().strip())
                elif "<number of stations>" in row:
                    raise NotImplementedError
                elif "<order strength>" in row:
                    self.OS = float(f.readline().strip())
                elif "<task times>" in row:
                    self.t = np.array([int(f.readline().strip().split(" ")[-1])
                                       for _ in range(self.N)])
                elif "<precedence relations>" in row:
                    self.P = np.zeros((self.N, self.N), dtype=bool)
                    for p in f.readlines():
                        if p.strip():
                            p = tuple(p.strip().split(","))
                            self.P[int(p[0]) - 1, int(p[-1]) - 1] = 1
                        else: break
                elif "<sequence dependent time increments>" in row:
                    raise NotImplementedError
                elif "<linked tasks>" in row:
                    self.LT = np.zeros((self.N, self.N), dtype=bool)
                    for p in f.readlines():
                        if p.strip():
                            p = tuple(p.strip().split(","))
                            self.LT[int(p[-1]) - 1, int(p[0]) - 1] = 1
                        else: break
                elif "<total station cost>" in row:
                    raise NotImplementedError
                elif "<station cost per unit>" in row:
                    raise NotImplementedError
                elif "<total task cost>" in row:
                    raise NotImplementedError
                elif " <task cost per unit>" in row:
                    raise NotImplementedError
                # Parallel stations are duplicates of some serial station such
                # that the local cycle time is a multiple of the global cycle
                # time.
                # The maximal number of times a station can be installed in
                # parallel
                elif "<maximum degree of parallelism>" in row:
                    self.max_parallelism = int(f.readline().strip())
                elif "<number of equipments>" in row:
                    self.n_equipments = int(f.readline().strip())
                elif "<equipments per task>" in row:
                    self.equipment = \
                        np.array([int(f.readline().strip().split(" ")[-1])
                                  for _ in range(self.N)])
                elif "<number of task attributes>" in row:
                    raise NotImplementedError
                elif "<task attribute values>" in row:
                    raise NotImplementedError
                elif "<attribute bounds per station>" in row:
                    raise NotImplementedError
                elif "<incompatible tasks>" in row:
                    raise NotImplementedError
                elif "<end>" in row:
                    break
                row = f.readline()

    def write_model(self) -> pyscipopt.Model:
        self._retrieve_data()
        model = self._write_model()
        return model

    @abc.abstractmethod
    def _retrieve_data():
        pass

    @abc.abstractmethod
    def _write_model(self):
        pass

if __name__ == '__main__':
    pass
