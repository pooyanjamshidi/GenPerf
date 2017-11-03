# -*- coding: utf-8 -*-

import unittest
from gp.genmodel import *
import os.path

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_evaluate_model(self):
        n = 8
        perf_model_txt = "2*o1 + 3*o1*o2 + 4*o2"
        perf_model = genModelfromString(perf_model_txt)
        startingModel = Model(perf_model)
        startingModel.name = "test"
        xTest = np.random.randint(2, size=(n, ndim))
        evaluate2csv(startingModel, xTest)
        if os.path.exists(startingModel.name + ".csv"):
            assert True
        else:
            assert False

    def test_absolute_truth_and_meaning(self):
        assert True


# if __name__ == '__main__':
#     unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(BasicTestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)