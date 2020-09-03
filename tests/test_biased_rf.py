import unittest
from barf.biased_rf import RandomForestClassifier as rfc
from barf.biased_rf import get_sorted_labels, BiasedRFClassifier, k_nearest_neighbor
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
        x_array = rfc()._get_val(x_df)
        np.testing.assert_array_equal(self.x, x_array)

    def test_get_val_with_list(self):
        x_list = self.x.tolist()
        x_array = rfc()._get_val(x_list)
        np.testing.assert_array_equal(self.x, x_array)

    def test_stack_df_and_array(self):
        x_df = pd.DataFrame(
            self.x, 
            columns=[f'col_{i}' for i in range(self.x.shape[1])]
        )
        combined = rfc._stack(x_df, self.y)
        np.testing.assert_array_equal(self.x, combined[:, :-1])
        np.testing.assert_array_equal(self.y.ravel(), combined[:, -1].ravel())

    def test_fit(self):
        model = rfc()
        model.fit(self.x, self.y)
        self.assertTrue(model._fitted)

    def test_predict(self):
        model = rfc()
        model.fit(self.x, self.y)
        pred = model.predict(self.x)
        unique_preds = np.unique(pred)
        unique_labels = np.unique(self.y)
        for p in unique_preds:
            with self.subTest(p=p):
                self.assertTrue(p in unique_labels)

    def test_get_sorted_labels(self):
        y = np.zeros((10, 1))
        y[:3] = 2
        y[3] = 1
        labels = get_sorted_labels(y)
        self.assertListEqual([1, 2, 0], labels)

    def test_k_nearest_neighbor(self):
        # create a set where there is a repeat of the first row
        x = np.concatenate([self.x, np.array([self.x[0, :]] * 5)], axis=0)
        neighbors = k_nearest_neighbor(x[0, :], x[1:, :], 5)
        # all five neighbors should be the newly added 5 rows, which is the point itself (0 distance)
        for i in range(5):
            with self.subTest(i=i):
                self.assertListEqual(x[0, :].tolist(), neighbors[i, :].tolist())
