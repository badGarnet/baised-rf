#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
import math

log = logging.getLogger(name=__name__)


def get_val(x):
    if isinstance(x, pd.DataFrame):
        x_val = x.values
    else:
        x_val = np.array(x)
    return x_val

def stack(x, y):
    x_val = get_val(x)
    y_val = get_val(y)
    if x_val.ndim != 2:
        raise ValueError(f'x must be a 2D array but got shape {x.shape}')
    if y_val.ndim == 1:
        y_val = y.reshape(-1, 1)
    elif y_val.ndim != 2:
        raise ValueError(f'y must be either a 1D or 2D array but got shape {y.shape}')
    return np.concatenate([x_val, y_val], axis=1)


def k_fold_split(data, folds=5, random_seed=None):
    np.random.seed(random_seed)
    np.random.shuffle(data)
    d_split = math.ceil(len(data) / folds)
    splits = list()
    for i in range(folds):
        if isinstance(data, pd.DataFrame):
            splits.append(data.iloc[
                i * d_split : min((i+1) * d_split , len(data))
            ])
        else:
            splits.append(data[
                i * d_split : min((i+1) * d_split , len(data))
            ])
    return (splits)


def prepare_k_fold_data(x, y=None, folds=5, random_seed=None):
    collection = list()
    if y is not None:
        data = stack(x, y)
    else:
        data = x
    idx_y_start = x.shape[1]
    splits = np.array(k_fold_split(data, folds, random_seed))
    for i, split in enumerate(splits):
        others = [j for j  in range(folds) if j != i]
        others = np.concatenate(splits[others], axis=0)
        if y is not None:
            # return x_train, x_test, y_train, y_test
            collection.append((
                others[:, :idx_y_start], split[:, :idx_y_start], 
                others[:, idx_y_start:], split[:, idx_y_start:], 
            ))
        else:
            # return train, test
            collection.append((others, split))
    return collection


def k_fold_validation(estimator, x, y, folds=5, random_seed=None, callbacks=None):
    """perform a k-fold validation on the ``estimator``; optionally return metrics for 
    each fold with functions listed in ``callbacks``

    Args:
        estimator ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]
        folds (int, optional): [description]. Defaults to 5.
        random_seed ([type], optional): [description]. Defaults to None.
        callbacks ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # using the same random seed to ensure shuffling still preserves x and y matching
    splits = prepare_k_fold_data(x, y, folds, random_seed)
    call_returns = dict()
    for i, split in enumerate(splits):
        trainx, valx, trainy, valy = split
        estimator = estimator.fit(trainx, trainy)
        pred_y = estimator.predict(valx)

        call_returns[f'fold_{i+1}'] = list()

        if callbacks is not None:
            for callback in callbacks:
                call_returns[f'fold_{i+1}'].append(callback(valy, pred_y))
    return call_returns


def roc_curve(y_true, y_score, pos_label=None):
    if pos_label is None:
        pos_label = 1
    classes = np.unique(y_true)

    # convert y_true to bools
    y_true = (y_true == pos_label)

    # sort the score and truth labels to be used later
    sorted_idx = np.argsort(y_score, kind='margesort')[::-1]
    y_score = y_score[sorted_idx]
    y_true = y_true[sorted_idx]

    # getting indices for distinct values from y_score for thresholds
    threshold_idxs = np.where(np.diff(y_score))[0]
    # add the last point so we have a curve reaching to the end
    threshold_idxs = np.append(threshold_idxs, [len(y_true)-1])
    # calculate at each threshold how many true positives are by summing 
    # y_true at each threshold level using cumsum
    tps = np.cumsum(y_true)[threshold_idxs]
    # calculate the false positive numbers at each threshold level
    # the threshold index itself is the count for predicted positive 
    # at each threshold (+1 for numpy starting at 0)
    fps = 1 + threshold_idxs - tps
    # make sure curves starts at 0, 0
    if (tps[0] != 0) or (fps[0] != 0):
        tps = np.append([0], tps)
        fps = np.append([0], fps)

    return fps, tps, y_score[threshold_idxs]




def classifier_report(y, pred):
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


def train_test_split(*args, test_size=0.3, strat=None, random_seed=None):
    # first shuffle the index selection
    idx = list(range(len(args[0])))
    np.random.seed(random_seed)
    np.random.shuffle(idx)
    # if strat is provided we have to split using the stratified column to ensure
    # each category gets split the same way
    if strat is None:
        split_index = int(len(args[0]) * test_size)
    else:
        split_index = int(len(args[0]) * test_size)
    
    results = list()
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            results.append(arg.iloc[idx[:-split_index]])
            results.append(arg.iloc[idx[-split_index:]])
        else:
            results.append(arg[idx[:-split_index]])
            results.append(arg[idx[-split_index:]])

    return (results)

class BaseEstimator:
    def __init__(self):
        self.name = 'BaseEstimator'

    @staticmethod
    def _get_val(x):
        # borrowing module function to be part of the object for easy inherience use
        return get_val(x)

    @staticmethod
    def _stack(x, y):
        # borrowing module function to be part of the object for easy inherience use
        return stack(x, y)

    @staticmethod
    def _parse_params(parms):
        return NotImplemented

    @staticmethod
    def _check_value(value, deep=False):
        if isinstance(value, BaseEstimator):
            return value.get_params(deep)
        else:
            return value

    @staticmethod
    def _flatten_param_dict(param_dict, collection, prefix=None):
        for key, value in param_dict.items():
            if isinstance(value, dict):
                collection = BaseEstimator._flatten_param_dict(value, collection, key)
            else:
                if prefix is None:
                    collection[key] = value
                else:
                    collection['__'.join([prefix, key])] = value
        return collection

    def get_params(self, deep=False):
        results = dict()
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                if deep:
                    results[key] = self._check_value(value, deep)
            else:
                results[key] = self._check_value(value)
        
        flattened = dict()
        return self._flatten_param_dict(results, flattened)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                log.warning(f'key {key} is not an attribute of the {self.__class__.__name__}')
