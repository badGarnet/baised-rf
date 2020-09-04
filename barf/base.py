#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(name=__name__)

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
    
    train = list()
    test = list()
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            train.append(arg.iloc[idx[:-split_index]])
            test.append(arg.iloc[idx[-split_index:]])
        else:
            train.append(arg[idx[:-split_index]])
            test.append(arg[idx[-split_index:]])

    return (train + test)

class BaseEstimator:
    def __init__(self):
        self.name = 'BaseEstimator'

    @staticmethod
    def _get_val(x):
        if isinstance(x, pd.DataFrame):
            x_val = x.values
        else:
            x_val = np.array(x)
        return x_val

    @staticmethod
    def _stack(x, y):
        x_val = BaseEstimator._get_val(x)
        y_val = BaseEstimator._get_val(y)
        if x_val.ndim != 2:
            raise ValueError(f'x must be a 2D array but got shape {x.shape}')
        if y_val.ndim == 1:
            y_val = y.reshape(-1, 1)
        elif y_val.ndim != 2:
            raise ValueError(f'y must be either a 1D or 2D array but got shape {y.shape}')
        return np.concatenate([x_val, y_val], axis=1)

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
