from .base import BaseEstimator
from .refrence_forest import build_tree, subsample, bagging_predict
import numpy as np
import pandas as pd


class BiasedRandomForestClassifier(BaseEstimator):
    def __init__(
        self, n_estimators=100, max_depth=10, random_seed=None,
        min_leaf_size=1, max_features=None, min_sample_split=2,
        sub_samples=1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.min_leaf_size = min_leaf_size
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.sub_samples = sub_samples
        self.name = 'BiasedRandomForestClassifier'
        self._trees = list()
        self._fitted = False

    @staticmethod
    def _get_val(x):
        if isinstance(x, pd.DataFrame):
            x_val = x.values
        else:
            x_val = np.array(x)
        return x_val

    @staticmethod
    def _stack(x, y):
        x_val = BiasedRandomForestClassifier._get_val(x)
        y_val = BiasedRandomForestClassifier._get_val(y)
        if x_val.ndim != 2:
            raise ValueError(f'x must be a 2D array but got shape {x.shape}')
        if y_val.ndim == 1:
            y_val = y.reshape(-1, 1)
        elif y_val.ndim != 2:
            raise ValueError(f'y must be either a 1D or 2D array but got shape {y.shape}')
        return np.concatenate([x_val, y_val], axis=1)

    def fit(self, x, y):
        self._trees = list()
        data = self._stack(x, y)
        if self.max_features is None:
            self.max_features = x.shape[1]
        for i in range(self.n_estimators):
            sample = subsample(data, self.sub_samples) 
            tree = build_tree(
                sample, self.max_depth, self.min_leaf_size,
                self.max_features
            )
            self._trees.append(tree)

        self._fitted = True

        return self

    def predict(self, x):
        if not self._fitted:
            raise ValueError(f"model must be fitted first by calling {self.__class__.__name__}.fit(x, y)")
        x_val = self._get_val(x)
        predictions = [
            bagging_predict(self._trees, row) for row in x_val
        ]
        return np.array(predictions)

    def report(self, x, y):
        pred = self.predict(x)
        tp = (y * pred).sum()
        tn = ((1 - y) * (1 - pred)).sum()
        fp = ((1 - y) * pred).sum()
        fn = (y * (1 - pred)).sum()
        return {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
