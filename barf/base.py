#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
import math

log = logging.getLogger(name=__name__)


def get_val(x):
    """helper function to get numpy array from the object

    Args:
        x (numpy.ndarray, pd.DataFrame, iterable): data to be converted into numpy.ndarray

    Returns:
        numpy.ndarray: numpy.ndarray representation of ``x``
    """
    if isinstance(x, pd.DataFrame):
        x_val = x.values
    else:
        x_val = np.array(x)
    return x_val

def stack(x, y):
    """Stacking x and y together along axis=1

    Args:
        x (numpy.ndarray or list): must be 2D
        y (numpy.ndarray or list): must either have the same length as ``x`` or the size of its first
            dimension must match that of ``x``

    Raises:
        ValueError: when x is not 2D
        ValueError: when y can't be appended to x

    Returns:
        numpy.ndarray: combined data with y appened as new columns to x
    """
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
    """splits data into folds randomly

    Args:
        data (numpy.ndarray): data to split, if more than 1D splits along first axis.
        folds (int, optional): number of folds to split. Defaults to 5.
        random_seed (int or None, optional): sets the random seed. Defaults to None.

    Returns:
        tuple: splitted data, each member is a portion from the input ``data``
    """
    # shuffle data
    np.random.seed(random_seed)
    np.random.shuffle(data)

    # sanity check
    if folds > len(data):
        raise ValueError(f"folds ({folds}) must be smaller than or equal to the number of data points ({len(data)}")

    # compute the size for each split
    d_split = math.ceil(len(data) / folds)
    splits = list()
    for i in range(folds):
        # add contents of the same size to each split
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
    """prepares data (either in one data object or separated x and y) into folds; 
    each fold contains a training set and a validation set. If ``y`` is not ``None``
    each fold contains train_x, test_x, train_y, test_y, in that order

    Args:
        x (numpy.ndarray or list): must be 2D, could contain target when ``y`` is ``None``
        y (numpy.ndarray or list):, optional): target column. Defaults to None.
        folds (int, optional): number of validation folds to generate. Defaults to 5.
        random_seed (int or None, optional): set the random seed. Defaults to None.

    Returns:
        list: each memnber is one fold of data, see above for details
    """
    collection = list()
    if y is not None:
        data = stack(x, y)
    else:
        data = x

    # record where to break x an y when needed
    idx_y_start = x.shape[1]

    # split the data into n groups
    splits = np.array(k_fold_split(data, folds, random_seed))

    # use each group as a test set and for each group combine the rest of 
    # the data as the corresponding train set
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
        callbacks (dict, optional): keys specify what result to use ('labels' or 'probs'). Defaults to None.

    Returns:
        [type]: [description]
    """
    # using the same random seed to ensure shuffling still preserves x and y matching
    splits = prepare_k_fold_data(x, y, folds, random_seed)
    call_returns = dict()
    for i, split in enumerate(splits):
        trainx, valx, trainy, valy = split
        estimator = estimator.fit(trainx, trainy)
        pred_y = estimator.predict(valx, return_prob=False)
        pred_p = estimator.predict(valx, return_prob=True)
        preds = {'labels': pred_y, 'probs': pred_p}

        call_returns[f'fold_{i+1}'] = dict()

        if callbacks is not None:
            for key, callback in callbacks.items():
                call_returns[f'fold_{i+1}'][key] = (
                    callback[1](valy, preds[callback[0]]))
    return call_returns


def get_tps_fps(y_true, y_score, pos_label=None):
    """generate true positive rate and false positive rate with different thresholds

    Args:
        y_true (numpy.ndarray): labels
        y_score (numpy.ndarray): predicted probability, must have the same shape as ``y_true``
        pos_label (optional): label for the positive case in binary classification. 
            Defaults to None (which means number 1 is positive)

    Returns:
        numpy.ndarray: true positive rate at a given threshold
        numpy.ndarray: false positive rate at a given threshold
    """
    if pos_label is None:
        pos_label = 1

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
    fps = 1 + threshold_idxs - tps
    return tps, fps


def prc_curve(y_true, y_score, pos_label=None):
    """make precision-recall curve for classification probability scores.
    Source `Suzanne Ekelund <https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used>`_

    Args:
        y_true (numpy.ndarray): labels
        y_score (numpy.ndarray): predicted probability, must have the same shape as ``y_true``
        pos_label (optional): label for the positive case in binary classification. 
            Defaults to None (which means number 1 is positive)]

    Returns:
        numpy.ndarray: true positive rate at a given threshold
        numpy.ndarray: false positive rate at a given threshol
    """
    tps, fps = get_tps_fps(y_true, y_score, pos_label)
    recall, precision = tps / y_true.sum(), tps / (tps + fps)
    if (recall[0] != 0) or (precision[0] != 1):
        recall = np.append([0], recall)
        precision = np.append([1], precision)
    return recall, precision


def roc_curve(y_true, y_score, pos_label=None):
    """generates the reciever operating characteristics curves for labels ``y_true``
    and predicted probability ``y_score``. 
    Source `wiki <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    Args:
        y_true (numpy.ndarray): labels
        y_score (numpy.ndarray): predicted probability, must have the same shape as ``y_true``
        pos_label (optional): label for the positive case in binary classification. 
            Defaults to None (which means number 1 is positive)

    Returns:
        numpy.ndarray: true positive rate at a given threshold
        numpy.ndarray: false positive rate at a given threshold

    """
    tps, fps = get_tps_fps(y_true, y_score, pos_label)
    # the threshold index itself is the count for predicted positive 
    # at each threshold (+1 for numpy starting at 0)
    if (tps[0] != 0) or (fps[0] != 0):
        tps = np.append([0], tps)
        fps = np.append([0], fps)
    return fps / (len(y_true) - y_true.sum()), tps / y_true.sum()


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
        # use '__' to unravel nested dict as a flat dict
        # e.g., {'a': 1, 'b': {'a': 1, 'b': 0}} ==> {'a': 1, 'b__a':1, 'b__b':0}
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
        """get a summary of estimator parameters recursively

        Args:
            deep (bool, optional): If True, private parameters will be exposed as well. Defaults to False.

        Returns:
            dict: estimator parameters in dict format; if a parameter named 'foo' is a ``BaseEstimator`` object,
                its params will be returned as 'foo__<param_name>'
        """
        results = dict()
        for key, value in self.__dict__.items():
            # check if an attribute is a protected attribute
            if key.startswith('_'):
                # only expose it if deep==True
                if deep:
                    # use check_values in case value is an BaseEstimator
                    results[key] = self._check_value(value, deep)
            else:
                # use check_values in case value is an BaseEstimator
                results[key] = self._check_value(value)
        
        flattened = dict()
        return self._flatten_param_dict(results, flattened)

    def set_params(self, **kwargs):
        """set the parameters of the estimator recursively; for nested objects use 
        '<attribute_name>__<nested_object_param_name>' to reach them.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                obj = getattr(self, key)
                if isinstance(obj, BaseEstimator):
                    # strip the header of the keys
                    new_values = {'__'.join(key.split('__')[1:]): val for key, val in value.items()}
                    # set the param of the nested estimator
                    obj.set_params(**new_values)
                else:
                    setattr(self, key, value)
            else:
                log.warning(f'key {key} is not an attribute of the {self.__class__.__name__}')
