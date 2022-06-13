from unittest import TestCase
import random
import numpy as np
import pandas as pd
from pandas.api.types import (is_integer_dtype, is_float_dtype, is_string_dtype)
from rex.preprocessing import PreprocessPipeline, Map
from test.BaseTest import BaseTest
from test.base_datasets_test import BaseDatasetsTest
from rex.tools import check_weights_dataframe, check_features


class DatasetTest(BaseTest):

    def test_not_a_dataframe_error(self):
        self._exception_test("not a dataframe")

    def test_wrong_sizes_error(self):
        too_short_dataframe = pd.DataFrame({
            'column_one': []
        })
        self._exception_test(too_short_dataframe)

    def test_nan_values_error(self):
        nan_dataset = pd.DataFrame({
            'feature_id': [np.nan, 2],
            'feature_1': [2, np.nan],
            'feature_2': range(2)
        })
        self._exception_test(nan_dataset)

    def test_duplicates_error(self):
        duplicates = pd.DataFrame({
            'feature_id': [1, 1],
            'feature_1': [2, 2],
        })
        self._exception_test(duplicates)

    def _exception_test(self, dataset):
        with self.assertRaises(ValueError):
            check_features(dataset)
