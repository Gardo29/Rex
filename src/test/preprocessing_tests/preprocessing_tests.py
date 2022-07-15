import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal  # <-- for testing dataframes
from pandas.testing import assert_series_equal
from rex.preprocessing import *


class PipelineTests(unittest.TestCase):
    """ class for running unittests"""

    def setUp(self):
        self.resources = '../../../resources'
        self.ratings = '/ml-small/ratings.csv'
        self.movie_features = '/ml-small/movies.csv'
        self.tag_file = '/ml-small/tags.csv'
        try:
            self.data = pd.read_csv(self.resources + self.ratings)
            self.features = pd.read_csv(self.resources + self.movie_features)
            self.tags = pd.read_csv(self.resources + self.tag_file)
        except IOError as e:
            print(e)
        # Adding dataframe and series equality assertions to the assertEqual function
        self.addTypeEqualityFunc(pd.DataFrame, self._assertDataFrameEqual)
        self.addTypeEqualityFunc(pd.Series, self._assertSeriesEqual)
        self.addTypeEqualityFunc(PreprocessedDataFrame, self._assertPreprocessedDataFrameEqual)

    # equality supports
    def _assertPreprocessedDataFrameEqual(self, a, b, msg):
        self._assertDataFrameEqual(a.dataframe, b.dataframe, msg)

    def _assertDataFrameEqual(self, a, b, msg):
        try:
            assert_frame_equal(a, b, check_dtype=False)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def _assertSeriesEqual(self, a, b, msg):
        try:
            assert_series_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def test_select_correct_output(self):
        select_feature_df = Select('timestamp').fit_transform(self.data)
        expected_output = pd.DataFrame(self.data['timestamp'])
        self.assertEqual(expected_output, select_feature_df.dataframe)
        print('Select test succeeded')

    def test_select_multiple_features(self):
        features = ['timestamp', 'rating']
        select_features_df = Select(features).fit_transform(self.data)
        expected_output = self.data[features]
        self.assertEqual(expected_output, select_features_df.dataframe)
        print('Select multiple features test succeeded')

    def test_correct_clip_output(self):
        low = 0
        high = 5
        feature = 'timestamp'
        clip_features_df = Clip(feature, lower=low, upper=high).fit_transform(self.data)
        self.data[feature] = self.data[feature].clip(lower=low, upper=high)
        self.assertEqual(self.data, clip_features_df.dataframe)
        print('Clip test succeeded')

    def test_check_negative_float_clip(self):
        low = -4
        high = 6
        feature = 'rating'
        self.data = Map(feature, lambda x: x - 3).fit_transform(self.data).dataframe
        clip_negative_df = Clip(feature, lower=low, upper=high).fit_transform(self.data)
        self.data[feature] = self.data[feature].clip(lower=low, upper=high)
        self.assertEqual(self.data, clip_negative_df.dataframe)

    def test_clip_raises_error_with_string_and_nan_values(self):
        low = -2
        high = 6
        feature = 'rating'
        for map_value in ['string', np.nan]:
            self.data = Map(feature, {3: map_value}).fit_transform(self.data)
            # Check when there are strings and NaN involved the exception TypeError is raised
            with self.assertRaises(TypeError):
                Clip(feature, lower=low, upper=high).fit_transform(self.data)

    def test_correct_drop_output(self):
        for feature in ['timestamp', ['timestamp']]:
            drop_df = Drop(feature).fit_transform(self.data)
            expected_output = self.data.drop(columns=feature)
            self.assertEqual(expected_output, drop_df.dataframe)

    def test_map(self):
        feature = 'rating'
        # lambda mapping
        lambda_df = Map(feature, lambda x: x - 1).fit_transform(self.data)
        self.data[feature] = self.data[feature].map(lambda x: x - 1)
        self.assertEqual(lambda_df.dataframe, self.data)
        # dict mapping
        map_df = Map(feature, {4: 'four'}).fit_transform(self.data)
        self.data[feature] = self.data[feature].map(lambda x: 'four' if x == 4 else x)
        self.assertEqual(map_df.dataframe, self.data)

    def test_correct_map_output(self):
        feature = 'rating'
        map_function = lambda x: x - 1
        map_df = Map(feature, map_function).fit_transform(self.data)
        self.data[feature] = self.data[feature].map(map_function)
        self.assertEqual(self.data, map_df.dataframe)
        print('Map test succeeded')

    def test_map_illegal_operation_on_string(self):
        feature = 'rating'
        mapped_dataframe = Map(feature, {3: 'string'}).fit_transform(self.data)
        # Check when there are strings and NaN involved the exception TypeError is raised
        with self.assertRaises(TypeError):
            Map(feature, lambda x: x - 1).fit_transform(mapped_dataframe)

    def test_fill_nan_methods(self):
        def fill_na_setup(feature, to_na, fill_value=None, method=None):
            dataset_2_nan = Map(feature, {to_na: np.nan}).fit_transform(self.data)
            fill_nan_values_df = FillNa(feature, fill_value, method=method).fit_transform(dataset_2_nan)
            return fill_nan_values_df, dataset_2_nan

        feature = 'rating'
        fill_na_value_df, dataset_to_nan = fill_na_setup(feature, 4, 10)
        self.assertEqual(fill_na_value_df.dataframe, dataset_to_nan.dataframe.fillna(10))
        # Check if there are no NaN values on the specific column
        check_for_nan = fill_na_value_df.dataframe[feature].isnull().values.any()
        self.assertFalse(check_for_nan)
        # ffill method test
        fill_na_value_df, dataset_to_nan = fill_na_setup(feature, 5, method='ffill')
        self.assertEqual(fill_na_value_df.dataframe, dataset_to_nan.dataframe.fillna(method='ffill'))
        check_for_nan = fill_na_value_df.dataframe[feature].isnull().values.any()
        self.assertFalse(check_for_nan)
        # bfill method test
        fill_na_value_df, dataset_to_nan = fill_na_setup(feature, 4, method='bfill')
        self.assertEqual(fill_na_value_df.dataframe, dataset_to_nan.dataframe.fillna(method='bfill'))
        check_for_nan = fill_na_value_df.dataframe[feature].isnull().values.any()
        self.assertFalse(check_for_nan)

    def test_fillna(self):
        feature = 'rating'
        fill_nan_values_df = PreprocessPipeline([Map(feature, {4: np.nan}),
                                                 FillNa(feature, 4)]).fit_transform(self.data)
        self.assertEqual(self.data, fill_nan_values_df.dataframe)

    # TODO
    def test_correct_min_max_scaler_output(self):
        self.data['rating'] = [1.0, 2.0, 3.0, 4.0, 5.0]
        minmax_preprocess = PreprocessPipeline([Drop('timestamp'),
                                                MinMaxScaler('userId', 'rating')])
        new_df = minmax_preprocess.fit_transform(self.data)
        expected_dict = {'userId': [1, 1, 1, 1, 1],
                         'movieId': [1, 3, 6, 47, 50],
                         'rating': [0.0, 0.25, 0.5, 0.75, 1.0]}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, new_df.dataframe)
        print('MinMaxScaler test succeeded')

    # TODO
    def test_correct_min_max_scaler_output_different_groupings(self):
        self.data['userId'] = [1, 1, 1, 4, 5]
        self.data['rating'] = [1.0, 2.0, 4.0, 4.0, 5.0]
        minmax_preprocess = PreprocessPipeline([Drop('timestamp'),
                                                MinMaxScaler('userId', 'rating')])
        new_df = minmax_preprocess.fit_transform(self.data)

        expected_dict = {'userId': [1, 1, 1, 4, 5],
                         'movieId': [1, 3, 6, 47, 50],
                         'rating': [0.0, 0.33, 1.0, 0.0, 0.0]}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, new_df.dataframe)

    def test_new_column_insertion(self):
        new_column_name = 'added'
        self.data[new_column_name] = 1
        for update_mode in [{new_column_name: 1}, self.data[new_column_name], self.data]:
            updated = Update(update_mode).fit_transform(self.data)
            self.assertEqual(self.data.columns.tolist(), updated.dataframe.columns.tolist())
            self.assertEqual(self.data, updated.dataframe)

    def test_correct_dropna_output(self):
        col_name = 'SomeNaNs'
        update_df = Update({col_name: pd.Series([1, 2, 3])}).fit_transform(self.data)
        drop_nans_df = DropNa(col_name).fit_transform(update_df)
        self.assertEqual(update_df.dataframe.dropna(), drop_nans_df.dataframe)

    def test_bin_correct_output(self):
        bins = 5
        baseline = 1
        new_df = Bin('rating', bins, baseline=1).fit_transform(self.data)
        self.data.rating = binned_statistic(self.data.rating, self.data.rating, bins=bins).binnumber + baseline
        self.assertEqual(self.data, new_df.dataframe)
        print('Bin test succeeded')

    # TODO
    def test_bin_threshold_correct_output(self):
        bin_threshold_preprocess = PreprocessPipeline([Drop('title'),
                                                       BinThreshold('genres', 'other', 7, divider='|')])
        new_df = bin_threshold_preprocess.fit_transform(self.features)
        new_df = new_df.dataframe.head()
        expected_dict = {'movieId': [1, 2, 3, 4, 5],
                         'genres': ['Adventure|Animation|Children|Comedy|Fantasy',
                                    'Adventure|Children|Fantasy',
                                    'Comedy|Romance', 'Comedy|Drama|Romance',
                                    'Comedy']}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, new_df)
        print('BinThreshold test succeeded')

    # TODO
    def test_bin_cumulative_correct_output(self):
        bin_cumulative_preprocess = PreprocessPipeline([Drop('title'),
                                                        BinCumulative('genres', 'other', 50, divider='|')])
        new_df = bin_cumulative_preprocess.fit_transform(self.features)
        new_df = new_df.dataframe.head()
        expected_dict = {'movieId': [1, 2, 3, 4, 5],
                         'genres': ['other', 'other',
                                    'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy']}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, new_df)
        print('BinCumulative test succeeded')

    def test_one_hot_encoding_single_feature(self):
        one_hot_encoding_preprocess = PreprocessPipeline([OneHotEncode('title')])
        cut_feature_df = self.features.head()
        new_df = one_hot_encoding_preprocess.fit_transform(cut_feature_df)
        expected_output = pd.concat([cut_feature_df.drop('title', axis=1), pd.get_dummies(cut_feature_df['title'])],
                                    axis=1)
        self.assertEqual(new_df.dataframe, expected_output)
        print(new_df)

    def test_one_hot_encoding_multiple_feature(self):
        feature = 'genres'
        one_hot_encoding_preprocess = PreprocessPipeline([OneHotEncode(feature, divider='|')])
        cut_feature_df = self.features.head()
        new_df = one_hot_encoding_preprocess.fit_transform(cut_feature_df)
        expected_output = pd.concat([cut_feature_df.drop(feature, axis=1),
                                     cut_feature_df[feature].str.get_dummies('|')],
                                    axis=1)
        self.assertEqual(new_df.dataframe, expected_output)
        print(new_df)

    def test_condense(self):
        separator = '|'
        feature = 'tag'
        id_col = 'movieId'
        self.tags['nans'] = np.nan
        self.tags['float_numbers'] = np.random.random(len(self.tags))
        self.tags = self.tags[[id_col, feature]]
        condensed_df = Condense(id_col, separator).fit_transform(self.tags)
        self.tags = self.tags.groupby(id_col).agg(lambda x: separator.join(x.astype(str).unique())).reset_index()
        self.assertEqual(condensed_df.dataframe, self.tags)

    # TODO
    def test_drop_duplicates(self):
        test_dataset = pd.DataFrame({
            'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
            'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
            'rating': [4, 4, 3.5, 15, 5]
        })
        drop_duplicates_df = DropDuplicates().fit_transform(test_dataset)
        expected_output = test_dataset.drop_duplicates(keep='last')
        self.assertEqual(expected_output, drop_duplicates_df.dataframe)
        # Single feature check
        drop_duplicates_df = DropDuplicates('brand', keep='first').fit_transform(test_dataset)
        expected_output = test_dataset.drop_duplicates(subset='brand', keep='first')
        self.assertEqual(drop_duplicates_df.dataframe, expected_output)
        # Multiple feature check
        drop_duplicates_df = DropDuplicates(['brand', 'style'], keep='first').fit_transform(test_dataset)
        expected_output = test_dataset.drop_duplicates(subset=['brand', 'style'], keep='first')
        self.assertEqual(drop_duplicates_df.dataframe, expected_output)
