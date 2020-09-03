#!/usr/bin/env python
import logging

log = logging.getLogger(name=__name__)

# base class definition for estimator
class BaseEstimator:
    def get_params(self, deep=False):
        return NotImplemented

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                log.warning(f'key {key} is not an attribute of the {self.__class__.__name__}')