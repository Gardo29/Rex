from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


class BaseTest(TestCase):
    def setUp(self) -> None:
        self.addTypeEqualityFunc(pd.DataFrame, self._assertDataframeEqual)
        self.addTypeEqualityFunc(pd.Series, self._assertSeriesEqual)
        self.addTypeEqualityFunc(np.ndarray, self._assertArraysEqual)

    def _assertDataframeEqual(self, a, b, msg):
        try:
            assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def _assertSeriesEqual(self, a, b, msg):
        try:
            assert_series_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    @staticmethod
    def _assertArraysEqual(a, b, msg):
        if not np.array_equal(a, b):
            raise AssertionError

    @staticmethod
    def print_result(dataset, test_function_name):
        print(test_function_name)
        print(dataset)

    def not_fallible_test(self, runnable):
        try:
            runnable()
        except:
            self.fail()
