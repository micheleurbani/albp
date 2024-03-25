import gzip
import ecole
import pickle
import numpy as np
from pyscipopt import Model
from pathlib import Path
from typing import Iterator

class ALBPGenerator:

    def __init__(self, directory: str,
                 sampling_mode: str = "remove_and_repeat",
                 rng: np.random.default_rng = np.random.default_rng(2985),
                 recursive: bool = True) -> None:
        """
        Create an iterator with the given parameters and a copy of the random
        state.

        Parameters
        ----------
        directory: str
            The directory path where to look for problem instances.

        recursive: bool
            Whether to recursively navigate folders, or look only to the
            provided folder.

        rng: ecole.RandomGenerator
            The Ecole random generator class to controll the process.range

        sampling_mode: str
            One of `replace`, `remove`, or `remove_and_repeat`.

                - `replace` replaces every file in the sampling pool right
                after it is sampled.
                - `remove` removes every file from the sampling pool right
                after it is sampled and finish iteration when all files have
                been sampled once.
                - `remove_and_repeat` remove every file from the sampling pool
                right after it is sampled but repeat the procedure (with
                different order) after all files have been sampled.
        """
        self.directory = Path(directory)
        # retrieve only lp files so far
        self.file_gen = self.directory.glob("**/*.lp")
        if not recursive:
            self.file_gen = self.directory.glob("*.lp")
        self.file_gen = list(self.file_gen)
        self.rng = rng
        self.mask = np.ones(len(self.file_gen), dtype=bool)
        self.sampling_mode = sampling_mode

    def __next__(self) -> ecole.scip.Model:
        if self.sampling_mode == "replace":
            i = self.rng.integers(len(self.file_gen))
        elif self.sampling_mode == "remove":
            if np.all(~self.mask):
                raise StopIteration
            i = self.rng.choice(np.arange(len(self.file_gen))[self.mask])
            self.mask[i] = 0
        elif self.sampling_mode == "remove_and_repeat":
            if np.all(~self.mask):
                self.mask = np.zeros(len(self.file_gen), dtype=bool)
            i = self.rng.choice(np.arange(len(self.file_gen))[self.mask])
            self.mask[i] = 0
        # file path of the next model to be load
        f_path = self.file_gen[i]
        # data path of the next model to be load
        d_path = f_path.parent.joinpath(f_path.stem)
        # create scip model
        model = Model()
        model.readProblem(str(f_path))
        # retrieve problem data
        with gzip.open(d_path, "rb") as f:
            data = pickle.load(f)
        # append data to the model
        model.data = data
        return model

    def __iter__(self) -> Iterator[ecole.scip.Model]:
        """Return itself as an iterator."""
        return self

    def seed(self, int) -> None:
        """Seed the random generator of the class."""
        self.rng.seed = int
