from .base import BaseEstimator
from .refrence_forest import build_tree, subsample, bagging_predict
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


class RandomForestClassifier(BaseEstimator):
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
        self.name = 'RandomForestClassifier'
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
        x_val = RandomForestClassifier._get_val(x)
        y_val = RandomForestClassifier._get_val(y)
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


def get_sorted_labels(y):
    """get unique labels and their counts from input 1D array ``y``; returns will be
    sorted by the counts

    Args:
        y (numpy.ndarray or list): must be 1D

    Returns:
        dict: dictionary containing the unique labels found in ``y`` as keys and
            the count for each label as values
    """
    y_labels, label_counts = np.unique(y, return_counts=True)
    labels = dict(zip(y_labels, label_counts))
    # sorting labels by counts
    labels = {k: v for k, v in sorted(labels.items(), key=lambda item: item[1])}
    return list(labels.keys())


def k_nearest_neighbor(p, candidates, n_neighbors):
    """get the ``n_neighbor`` nearest members from ``array`` to ``p``, measured
    by Euclidean distance

    Args:
        p (numpy.array or list): point from where to look for neighbors, must be 1D
        candidates (numpy.array or list): the domain from where we search for ``p``'s neighbors; must be 2D and the
            size of the second dimension matches the length of ``p``
        n_neighbors (int): number of neighbors to return
    """
    array = np.array(candidates)
    assert len(p) == array.shape[1]
    # calculate Euclidian distances
    distance = np.sqrt(np.sum((array - p) ** 2, axis=1))
    # return top neighbors
    sorted_index = np.argsort(distance)
    if len(distance) < n_neighbors:
        log.warning(f'requested {n_neighbors} neighbors but only have {len(distance)} candidates')
        return array
    else:
        return array[sorted_index, :]


class BiasedRFClassifier(BaseEstimator):
    def __init__(self, p_critical=0.5, k_nearest_neighbor=10, n_estimators=100):
        self.p_critical = p_critical
        self.k_nearest_neighbor = k_nearest_neighbor
        self.n_estimators = n_estimators
    
    def fit(self, x, y):
        # getting unique labels in y and their counts to figure out whta is majority
        labels_min, labels_maj =  get_sorted_labels(y)
        # get the row indices for major and minor labels
        index_maj = y[y==labels_maj]
        index_min = y[y==labels_min]
        # separate training set for major and minor labels
        x_maj = x[index_maj, :]
        x_min = x[index_min, :]

        for x in x_min:
            t_nn_x = neareast_neighbor(x, x_maj, self.k_nearest_neighbor)

        return NotImplemented
