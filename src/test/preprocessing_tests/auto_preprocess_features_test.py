from typing import Callable

import pandas as pd

from rex.preprocessing2 import auto_preprocess_features_dataframe, OneHotEncode, BinCumulative, PreprocessPipeline, \
    Select, DropDuplicates, DropNa, MinMaxScaler, PreprocessedDataFrame
from test.base_datasets_test import BaseDatasetsTest
import numpy as np

from test.check_tests.base_dataframes_test import BaseCheckTest


class AutoPreprocessFeaturesTest(BaseCheckTest):
    def _function(self) -> Callable:
        return auto_preprocess_features_dataframe

    def setUp(self) -> None:
        super(AutoPreprocessFeaturesTest, self).setUp()

    def test_not_dataframe_error(self):
        self._exception_test('not a dataframe')

    def test_wrong_size_dataframe_error(self):
        self._exception_test(self._item_features.iloc[:, 0])

    def test_size_dataframe_ok(self):
        self.not_fallible_test(lambda: self._function()(self._item_features))

    def test_check_if_not_preprocessed_gives_empty_pipeline(self):
        self._test(self._create_df([0.3, 0.5, 0.6]), [], pd.DataFrame)

    def test_drop_duplicates(self):
        dataframe = pd.DataFrame({
            'user_id': [1, 1, 2],
            'feature': [0.3, 0.5, 0.6]})
        self._test(dataframe, DropDuplicates(subset_features='user_id'), PreprocessedDataFrame)

    def test_nan_values(self):
        self._test(self._create_df([np.nan, 0, np.nan]), DropNa(), PreprocessedDataFrame)

    def test_min_max_scale_int_feature(self):
        self._test(self._create_df([3, 4, 5]), MinMaxScaler(features='feature'), PreprocessedDataFrame)

    def test_min_max_scale_float_feature(self):
        self._test(self._create_df([3.4, 4.1, 5.3]), MinMaxScaler(features='feature'), PreprocessedDataFrame)

    def test_min_max_scale_doesnt_change_0_1_feature(self):
        self._test(self._create_df([0, 0.5, 1]), [], pd.DataFrame)

    def test_one_hot_encode_multiple_features_with_cumulative_binning(self):
        self._test(self._create_df(['f1|f2', 'f2|f3|f4', 'f1|f3']), [
            BinCumulative('feature', 'Other', divider='|'),
            OneHotEncode('feature', divider='|')
        ], PreprocessedDataFrame)

    def _test(self, dataframe, transformations, dtype):
        transformations = np.atleast_1d(transformations)
        result, pipeline = self._function()(dataframe, True)
        expected = PreprocessPipeline(transformations).fit_transform(dataframe)
        self.assertIsInstance(result, dtype)
        self.assertEqual(pipeline, PreprocessPipeline(
            expected.preprocess_functions if isinstance(expected, PreprocessedDataFrame) else []))
        self.assertEqual(result, expected)

    @staticmethod
    def _create_df(feature_value):
        return pd.DataFrame({
            'user_id': range(3),
            'feature': feature_value
        })
