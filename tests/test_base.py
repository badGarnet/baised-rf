import unittest
import numpy as np
from barf.base import BaseEstimator, train_test_split, roc_curve
import matplotlib.pyplot as plt


class TestBaseUtils(unittest.TestCase):
    def test_train_test_split(self):
        x = np.random.rand(100).reshape(10, 10)
        x1, x2 = train_test_split(x)
        self.assertTrue(len(x), len(x1) + len(x2))

    def test_train_test_split_randomness(self):
        x = np.random.rand(100).reshape(10, 10)
        x1, x2 = train_test_split(x)
        newx = np.concatenate([x1, x2], axis=0)
        self.assertFalse((x == newx).all())

    def test_roc_curve_random(self):
        y = np.random.randint(0, 2, 100)
        pred = np.random.rand(100)
        fps, tps, _ = roc_curve(y, pred)
        plt.plot(fps, tps, '--')
        plt.savefig('tmp/roc_random.png')
        plt.close()

    def test_roc_curve_perfect_good(self):
        y = np.random.randint(0, 2, 100)
        fps, tps, _ = roc_curve(y, y)
        self.assertEqual(0, fps[1])
        self.assertEqual(y.sum(), tps[1])
        plt.plot(fps, tps, '--r')
        plt.savefig('tmp/roc_perfect.png')
        plt.close()

    def test_roc_curve_perfect_bad(self):
        y = np.random.randint(0, 2, 100)
        fps, tps, _ = roc_curve(y, 1-y)
        self.assertEqual(0, tps[1])
        self.assertEqual(100-y.sum(), fps[1])
        plt.plot(fps, tps, '--r')
        plt.savefig('tmp/roc_perfect_bad.png')
        plt.close()


class TestEstimator(unittest.TestCase):
    def test_init(self):
        estimator = BaseEstimator()
        self.assertEqual('BaseEstimator', estimator.__class__.__name__)

    def test_get_params(self):
        estimator = BaseEstimator()
        params = estimator.get_params()
        self.assertEqual({'name': 'BaseEstimator'}, params)

    def test_check_value(self):
        a = 'a'
        self.assertEqual(a, BaseEstimator._check_value(a))

    def test_flatten_param_dict(self):
        a = {'a': 'a', 'b': {'a': 'b1', 'b': 'b2'}}
        b = {'a': 'a', 'b__a': 'b1', 'b__b': 'b2'}
        flattened = dict()
        flattened = BaseEstimator()._flatten_param_dict(a, flattened, prefix=None)
        self.assertDictEqual(
            b, flattened
        )

    def test_check_value_with_estimator(self):
        a = BaseEstimator()
        self.assertEqual({'name': 'BaseEstimator'}, BaseEstimator._check_value(a))

    def test_set_params(self):
        a = BaseEstimator()
        a.set_params(name='new')
        self.assertEqual('new', a.name)

    def test_set_params_not_found(self):
        a = BaseEstimator()
        a.set_params(foo='new')
        self.assertFalse(hasattr(a, 'foo'))

    def test_deep_params_false(self):
        class NewEstimator(BaseEstimator):
            def __init__(self, foo):
                self.estimator = BaseEstimator()
                self.foo = foo
                self.name = 'NewEstimator'
                self._hidden = 'secret'

        a = NewEstimator('foo')
        self.assertEqual(
            {'foo': 'foo', 'estimator__name': 'BaseEstimator', 'name': 'NewEstimator'},
            a.get_params()
        )

    def test_deep_params_true(self):
        class NewEstimator(BaseEstimator):
            def __init__(self, foo):
                self.estimator = BaseEstimator()
                self.foo = foo
                self.name = 'NewEstimator'
                self._hidden = 'secret'

        a = NewEstimator('foo')
        self.assertEqual(
            {'foo': 'foo', 'estimator__name': 'BaseEstimator', 'name': 'NewEstimator', '_hidden': 'secret'},
            a.get_params(deep=True)
        )

    def test_nesting_estimator(self):
        class NewEstimator(BaseEstimator):
            def __init__(self, foo):
                self.estimator = BaseEstimator()
                self.foo = foo
                self.name = 'NewEstimator'

        a = NewEstimator('foo')
        self.assertEqual(
            {'foo': 'foo', 'estimator__name': 'BaseEstimator', 'name': 'NewEstimator'},
            a.get_params()
        )
