import unittest
from pyscipopt import Model, SCIP_RESULT

from albp.data.reader import ALBReader


class TestALBReader(unittest.TestCase):

    def setUp(self) -> None:
        self.filename = "data/raw/albp-datasets/SALBP-1993/BOWMAN8.alb"

    def test_reader(self):
        m = Model("testALB")
        reader = ALBReader()
        m.includeReader(reader, "albpreader", "PyReader for ALB problems.", "alb")
        m.readProblem(self.filename)
        m.optimize()
        print(m.data[-1].N)
        self.assertTrue(m.getStatus() == 'optimal')