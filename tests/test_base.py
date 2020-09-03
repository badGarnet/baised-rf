import unittest
from barf.base import BaseEstimator


class TestEstimator(unittest.TestCase):
    def test_init(self):
        estimator = BaseEstimator
        self.assertEqual('BaseEstimator', estimator.__class__.__name__)