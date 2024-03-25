import unittest
import numpy as np
from pathlib import Path
from pyscipopt import Model

from models.generator.albp_gen import ALBPGenerator


class TestALBPGenerator(unittest.TestCase):

    def setUp(self) -> None:
        self.path = Path("data", "processed")

    def test_replace(self):
        self.gen = ALBPGenerator(
            directory=str(self.path),
            sampling_mode="replace",
            rng=np.random.default_rng(seed=3298)
        )
        model = next(self.gen)
        self.assertIsInstance(model, Model)

    def test_remove(self):
        self.gen = ALBPGenerator(
            directory=str(self.path),
            sampling_mode="remove",
            rng=np.random.default_rng(seed=5739)
        )
        model = next(self.gen)
        self.assertIsInstance(model, Model)

    def test_remove_and_repeat(self):
        self.gen = ALBPGenerator(
            directory=str(self.path),
            sampling_mode="remove_and_repeat",
            rng=np.random.default_rng(seed=4290)
        )
        model = next(self.gen)
        self.assertIsInstance(model, Model)


if __name__ == "__main__":
    unittest.main()
