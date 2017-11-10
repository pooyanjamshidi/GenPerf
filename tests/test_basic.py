# -*- coding: utf-8 -*-

import unittest
from gp.genmodel import *
import os.path

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_evaluate_model_2csv(self):
        n = 8
        ndim = 4
        perf_model_txt = "2*o1 + 3*o1*o2 + 4*o2"
        perf_model = genModelfromString(perf_model_txt)
        model = Model(perf_model, ndim=ndim)
        model.name = "test"
        xTest = np.random.randint(2, size=(n, ndim))
        evaluate2csv(model, xTest)
        if os.path.exists(model.name + ".csv"):
            assert True
        else:
            assert False

    def test_evaluate_model_fast(self):
        ndim = 4
        perf_model_txt = "2*o1 + 3*o1*o2 + 4*o2"
        perf_model = genModelfromString(perf_model_txt)
        model = Model(perf_model, ndim=ndim)
        model.name = "test_fast"
        xTest = np.array([[0, 1, 1, 0]])
        yTest = model.evaluateModelFast(xTest)
        if yTest == 2 + 3 + 4:
            assert  True
        else:
            assert False

    def test_evaluate_model_fast_on_large_model(self):
        ndim = 20
        perf_model_txt = "21 + 2.1*o1 + 4.2*o2 + 0.1*o3 + 100*o4 + 2*o5 + 0.1*o6 + o7 + o8 + o9 + o10 + 23*o1*o3 + 2*o4*o7 + o8*o9*o10"
        perf_model = genModelfromString(perf_model_txt)
        model = Model(perf_model, ndim=ndim)
        model.name = "test_fast"
        xTest = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        yTest = model.evaluateModelFast(xTest)
        if yTest == 21 + 2.1 + 4.2:
            assert  True
        else:
            assert False

    def test_evaluate_model_both(self):
        ndim = 20
        perf_model_txt = "21 + 2.1*o1 + 4.2*o2 + 0.1*o3 + 100*o4 + 2*o5 + 0.1*o6 + o7 + o8 + o9 + o10 + 23*o1*o3 + 2*o4*o7 + o8*o9*o10"
        perf_model = genModelfromString(perf_model_txt)
        model = Model(perf_model, ndim=ndim)
        model.name = "test_fast"
        xTest = np.random.randint(2, size=(1, ndim))
        yTest1 = model.evaluateModelFast(xTest)
        yTest2 = model.evaluateModel(xTest)

        if yTest1 == yTest2:
            assert  True
        else:
            assert False

    def test_absolute_truth_and_meaning(self):
        assert True


# if __name__ == '__main__':
#     unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(BasicTestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)