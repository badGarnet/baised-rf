import unittest
from barf.biased_rf import BiasedRandomForestClassifier as brfc
import numpy as np
import pandas as pd


class TestBiasedRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(100).reshape(10, 10)
        self.y = np.random.randint(0, 4, 10)

    def test_get_val(self):
        x_df = pd.DataFrame(
            self.x, 
            columns=[f'col_{i}' for i in range(self.x.shape[1])]
        )
        x_array = brfc()._get_val(x_df)
        np.testing.assert_array_equal(self.x, x_array)

    def test_get_val_with_list(self):
        x_list = self.x.tolist()
        x_array = brfc()._get_val(x_list)
        np.testing.assert_array_equal(self.x, x_array)

    def test_stack_df_and_array(self):
        x_df = pd.DataFrame(
            self.x, 
            columns=[f'col_{i}' for i in range(self.x.shape[1])]
        )
        combined = brfc._stack(x_df, self.y)
        np.testing.assert_array_equal(self.x, combined[:, :-1])
        np.testing.assert_array_equal(self.y.ravel(), combined[:, -1].ravel())

    def test_fit(self):
        model = brfc()
        model.fit(self.x, self.y)
        self.assertTrue(model._fitted)
