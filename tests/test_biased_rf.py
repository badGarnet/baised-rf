import unittest
from barf.biased_rf import BiasedRandomForestClassifier as brfc
import numpy as np
import pandas as pd


class TestBiasedRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        x = np.random.rand(0, 1, 100).reshape(10, 10)
        y = np.random.randint(0, 4, 10)

    def test_get_val(self):
        x_df = pd.DataFrame(
            self.x, 
            columns=[f'col_{i}' for i in range(self.x.shape[1])]
        )
        x_array = brfc()._get_val(x_df)
        np.testing.assert_array_equal(self.x, x_array)
