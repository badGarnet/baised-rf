import unittest
from barf.biased_rf import RandomForestClassifier as rfc
from barf.biased_rf import get_sorted_labels, BiasedRFClassifier, k_nearest_neighbor
import numpy as np
import pandas as pd


class TestBiasedRandomForestClassifier(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(100).reshape(10, 10)
        self.y = np.random.randint(0, 4, 10)
        self.imbalance_y = np.zeros_like(self.y)
        self.imbalance_y[:2] = 1

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

    def test_fit_multiprocessing(self):
        model = rfc()
        model.fit(self.x, self.y, multi=True)
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

    def test_k_nearest_neighbor_index(self):
        # create a set where there is a repeat of the first row
        x = np.concatenate([self.x, np.array([self.x[0, :]] * 5)], axis=0)
        neighbors = k_nearest_neighbor(x[0, :], x[1:, :], 5, return_index=True)
        # all five neighbors should be the newly added 5 rows, which is the point itself (0 distance)
        self.assertListEqual(list(range(9, 14)), neighbors)

    def test_get_critical_x_shape(self):
        x = np.concatenate([self.x, np.array([self.x[0, :]] * 5)], axis=0)
        y = np.zeros((len(x), 1))
        y[0] = 1
        critical_x, _ = BiasedRFClassifier(k_nearest_neighbor=5)._get_critical_set(x, y)
        # confirm getting the single minor and 5 cloeset => total 6 rows back
        self.assertEqual(6, critical_x.shape[0])
        self.assertEqual(x.shape[1], critical_x.shape[1])

    def test_get_critical_x_shape(self):
        x = np.concatenate([self.x, np.array([self.x[0, :]] * 5)], axis=0)
        y = np.zeros((len(x), 1))
        y[0] = 1
        critical_x, _ = BiasedRFClassifier(k_nearest_neighbor=5)._get_critical_set(x, y)
        # confirm getting the single minor and 5 cloeset => total 6 rows back
        self.assertEqual(6, critical_x.shape[0])
        self.assertEqual(x.shape[1], critical_x.shape[1])

    def test_get_critical_x_shape(self):
        x = np.concatenate([self.x, np.array([self.x[0, :]] * 5)], axis=0)
        y = np.zeros((len(x), 1))
        y[0] = 1
        _, critical_y = BiasedRFClassifier(k_nearest_neighbor=5)._get_critical_set(x, y)
        # confirm the critical y is the minor (label=1) and 5 major (label=0) stacked
        self.assertListEqual([1,0,0,0,0,0], critical_y.tolist())

    def test_get_critical_set_indices_single(self):
        x = np.concatenate([self.x, np.array([self.x[0, :]] * 5)], axis=0)
        x_min = x[:1, :]
        x_maj = x[1:, :]
        critical_set = BiasedRFClassifier(k_nearest_neighbor=5)._get_crtical_set_indices(x_maj, x_min)
        # confirm that the indices are the nearest neighbors
        self.assertEqual(list(range(9, 14)), critical_set)

    def test_get_critical_set_indices_overlap_neighbors(self):
        # create dataset where x_min are two points with the same features
        # so the critical set we get should just be 5 points (both have the same neighbors)
        self.x[1, :] = self.x[0, :]
        x_min = self.x[:2, :]
        x_maj = self.x[2:, :]
        critical_set = BiasedRFClassifier(k_nearest_neighbor=5)._get_crtical_set_indices(x_maj, x_min)
        # confirm that the indices are the nearest neighbors
        self.assertEqual(5, len(critical_set))

    def test_get_critical_set_indices_in_major_set(self):
        x_min = self.x[:2, :]
        x_maj = self.x[2:, :]
        critical_set = BiasedRFClassifier(k_nearest_neighbor=5)._get_crtical_set_indices(x_maj, x_min)
        # confirm that the indices are the nearest neighbors
        major_indices = list(range(len(x_maj)))
        for idx in critical_set:
            with self.subTest(idx=idx):
                self.assertTrue(idx in major_indices)

    def test_fit_logic(self):
        model = BiasedRFClassifier(k_nearest_neighbor=5)
        model.fit(self.x, self.imbalance_y)
        self.assertTrue(model._is_fitted())

    def test_predict_logic(self):
        model = BiasedRFClassifier(k_nearest_neighbor=5)
        model.fit(self.x, self.imbalance_y)
        pred = model.predict(self.x)
        self.assertEqual(len(self.imbalance_y), len(pred))
        print(pred)
