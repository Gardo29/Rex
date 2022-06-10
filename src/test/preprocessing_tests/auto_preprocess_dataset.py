from typing import Callable

import numpy as np
import pandas as pd
from rex.preprocessing2 import (PreprocessPipeline, auto_preprocess_weights_dataframe, Normalizer, Select, Map, DropNa,
                                DropDuplicates, ToCOOMatrix, PreprocessedDataFrame)
from rex.tools import get_df
from test.check_tests.base_dataframes_test import BaseCheckTest


class AutoPreprocessDatasetTest(BaseCheckTest):
    def _function(self) -> Callable:
        return auto_preprocess_weights_dataframe

    def setUp(self) -> None:
        super(AutoPreprocessDatasetTest, self).setUp()
        self._dataset = self._dataset.drop('timestamp', axis=1)

    def test_not_dataframe_error(self):
        self._exception_test('not a dataframe')

    def test_reverse_matrix_to_COO(self):
        dataframe = pd.DataFrame([
            ['good', 0, 0, 'very bad'],
            [0, 0, 'good', 'bad'],
            [0, 'very good', 'vary good', 'good']
        ])
        self._test(dataframe, [
            ToCOOMatrix('user_id', 'item_id', 'rating'),
            Map('rating'),
            Normalizer(group_by_features='user_id', features='rating')
        ], PreprocessedDataFrame, to_coo=True)

    def test_wrong_size_dataframe_error(self):
        self._exception_test(self._dataset.iloc[:, 0])
        self._exception_test(self._dataset.iloc[:, :1])

    def test_size_dataframe_ok(self):
        self.not_fallible_test(lambda: self._function()(self._dataset))

    def test_not_preprocessed_dataframe_empty_preprocessing(self):
        self._dataset['rating'] = 0
        self._test(self._dataset, [], pd.DataFrame)

    def test_too_much_columns_select(self):
        self._dataset.rating = 0
        self._dataset['4_column'] = 1
        self._test(self._dataset, [Select(['userId', 'movieId', 'rating'])], PreprocessedDataFrame)

    def test_drop_na(self):
        dataset = pd.DataFrame({
            'user_id': range(4),
            'item_id': range(4),
            'rating': [np.nan, 0, np.nan, 0]
        })
        self._test(dataset, [DropNa()], PreprocessedDataFrame)

    def test_drop_duplicates(self):
        dataset = pd.DataFrame({
            'user_id': [1, 2, 2],
            'item_id': [3, 4, 4],
            'rating': [0, 0, 0]
        })
        self._test(dataset, [DropDuplicates(subset_features=['user_id', 'item_id'])], PreprocessedDataFrame)

    def test_transform_categorical_weights(self):
        dataset = pd.DataFrame({
            'user_id': range(3),
            'item_id': range(3),
            'rating': list('abc')
        })
        self._test(dataset, [Map('rating'),
                             Normalizer(group_by_features='user_id', features='rating')], PreprocessedDataFrame)

    def test_l2_normalization_ratings(self):
        self._test(self._dataset, [Normalizer(group_by_features='userId', features='rating')], PreprocessedDataFrame)

    def _test(self, dataframe, transformations, dtype, to_coo=False):
        transformations = np.atleast_1d(transformations)
        result, pipeline = self._function()(dataframe, sparse_input=to_coo, return_pipeline=True)
        train, test, pipeline_split = self._function()(dataframe, sparse_input=to_coo, return_pipeline=True,
                                                       train_size=0.5)
        expected = PreprocessPipeline(transformations).fit_transform(dataframe)

        self.assertIsInstance(result, dtype)
        self.assertIsInstance(train, dtype)
        self.assertIsInstance(test, dtype)

        self.assertEqual(result, expected)
        self.assertEqual(pipeline, pipeline_split)
        self.assertEqual(pipeline, PreprocessPipeline([]) if isinstance(expected, pd.DataFrame) else PreprocessPipeline(
            expected.preprocess_functions))

        sort_by = get_df(expected).columns[:2].tolist()
        self.assertEqual(pd.concat([get_df(train), get_df(test)]).sort_values(by=sort_by),
                         get_df(expected).sort_values(by=sort_by))
