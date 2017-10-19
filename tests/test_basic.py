# -*- coding: utf-8 -*-

from gp.poly_parser import Parser

import unittest
from gp.func_util import function_utilities
from gp import poly_parser

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        assert True


# if __name__ == '__main__':
#     unittest.main()

suite = unittest.TestLoader().loadTestsFromTestCase(BasicTestSuite)
unittest.TextTestRunner(verbosity=2).run(suite)