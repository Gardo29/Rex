from unittest import TestCase
import random
import numpy as np
import pandas as pd
from pandas.api.types import (is_integer_dtype, is_float_dtype, is_string_dtype)
from rex.preprocessing import PreprocessPipeline, Map
from test.BaseTest import BaseTest
from test.base_datasets_test import BaseDatasetsTest
from rex.tools import check_weights


class DatasetTest(BaseTest):

    def test_not_a_dataframe_error(self):
        self._exception_test("not a dataframe")

    def test_wrong_sizes_error(self):
        too_short_dataset = pd.DataFrame({
            'column_one': []
        })
        too_long_dataset = pd.DataFrame({
            'column_one': [],
            'column_two': [],
            'column_three': [],
            'column_four': []
        })
        self._exception_test(too_short_dataset)
        self._exception_test(too_long_dataset)

    def test_weights_column_dtype_error(self):
        wrong_dtype_weights = pd.DataFrame({
            'user_id': range(3),
            'item_id': range(3),
            'weight': [str(num) for num in range(3)]
        })
        self.assertTrue(is_string_dtype(wrong_dtype_weights.weight.dtype))
        self._exception_test(wrong_dtype_weights)

    def test_nan_values_error(self):
        nan_dataset = pd.DataFrame({
            'user_id': [1, np.nan],
            'item_id': [2, np.nan],
            'weight': range(2)
        })
        self._exception_test(nan_dataset)

    def test_duplicates_error(self):
        duplicates = pd.DataFrame({
            'user_id': [1, 1],
            'item_id': [2, 2],
            'weight': range(2)
        })
        self._exception_test(duplicates)

    def test_ok_dataset_ok(self):
        integer_weights_dataset = pd.DataFrame({
            'user_id': range(2),
            'item_id': range(2),
            'weight': range(2)
        })
        float_weights_dataset = integer_weights_dataset.copy(deep=True)
        float_weights_dataset.weight = float_weights_dataset.weight.astype('float')

        self.assertTrue(is_integer_dtype(integer_weights_dataset.weight.dtype))
        self.not_fallible_test(lambda: check_weights(integer_weights_dataset))

        self.assertTrue(is_float_dtype(float_weights_dataset.weight.dtype))
        self.not_fallible_test(lambda: check_weights(float_weights_dataset))

    def _exception_test(self, dataset):
        with self.assertRaises(ValueError):
            check_weights(dataset)
