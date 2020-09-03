#!/usr/bin/env python
import logging

log = logging.getLogger(name=__name__)

# base class definition for estimator


class BaseEstimator:
    def __init__(self):
        self.name = 'BaseEstimator'

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
