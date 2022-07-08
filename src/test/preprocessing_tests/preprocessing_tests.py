import unittest

from pandas.testing import assert_frame_equal  # <-- for testing dataframes
from pandas.testing import assert_series_equal
from rex.preprocessing2 import *


class PipelineTests(unittest.TestCase):
    """ class for running unittests"""
    def setUp(self):
        self.test_input_dir = '../../../resources'
        self.test_file_dir = '/ml-small/ratings.csv'
        self.test_file_dir2 = '/ml-small/movies.csv'
        self.test_file_dir3 = '/ml-small/tags.csv'
        try:
            data = pd.read_csv(self.test_input_dir + self.test_file_dir)
            features = pd.read_csv(self.test_input_dir + self.test_file_dir2)
            tags = pd.read_csv(self.test_input_dir + self.test_file_dir3)
            # Take first 5 rows only
            self.data = data.head(5)
            # Full dataset
            self.features = features
            self.tags = tags
        except IOError as e:
            print(e)
        # Adding dataframe and series equality assertions to the assertEqual function
        self.addTypeEqualityFunc(pd.DataFrame, self._assertDataframeEqual)
        self.addTypeEqualityFunc(pd.Series, self._assertSeriesEqual)

    # equality supports
    def _assertDataframeEqual(self, a, b, msg):
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
        feature = 'timestamp'
        self.data[feature] = [9.0, -3.0, 0.0, -9.0, 7.0]
        clip_negative_df = Clip(feature, lower=low, upper=high).fit_transform(self.data)
        self.data[feature] = self.data[feature].clip(lower=low, upper=high)
        self.assertEqual(self.data, clip_negative_df.dataframe)
        print('Negative float clip test succeeded')

    def test_clip_with_string_and_nan_values(self):
        low = -2
        high = 6
        feature = 'timestamp'
        self.data[feature] = [9.0, -3.0, 'not registered', 'NaN', np.nan]
        # Check when there are strings and NaN involved the exception TypeError is raised
        with self.assertRaises(TypeError):
            Clip(feature, lower=low, upper=high).fit_transform(self.data)

        print('Clip with strings and NaN values test succeeded')

    def test_correct_drop_output(self):
        feature = 'timestamp'
        drop_df = Drop(feature).fit_transform(self.data)
        expected_output = self.data.drop(columns=feature)
        self.assertEqual(expected_output, drop_df.dataframe)
        print('Drop test succeeded')

    def test_map_using_dict(self):
        feature = 'rating'
        to_be_replaced = {4: 'four', 5: np.nan}
        map_df = Map(feature, to_be_replaced).fit_transform(self.data)
        self.data[feature] = self.data[feature].map(to_be_replaced)
        self.assertEqual(self.data, map_df.dataframe)
        print('Map test with dict succeeded')

    def test_correct_map_output(self):
        feature = 'rating'
        map_function = lambda x: x - 1
        map_df = Map(feature, map_function).fit_transform(self.data)
        self.data[feature] = self.data[feature].map(map_function)
        self.assertEqual(self.data, map_df.dataframe)
        print('Map test succeeded')

    def test_map_illegal_operation_on_string_and_nan_values(self):
        feature = 'timestamp'
        self.data[feature] = [9.0, -3, 'not registered', 'NaN', np.nan]
        # Check when there are strings and NaN involved the exception TypeError is raised
        with self.assertRaises(TypeError):
            Map(feature, lambda x: x - 1).fit_transform(self.data)
        print('Map with strings and NaN values test succeeded')

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

        print('FillNa with several methods test succeeded')

    def test_check_different_dtypes_fillna(self):
        feature = 'rating'
        dataset_2_different_dtypes = Map(feature, {5: 'ciao', 4: np.nan}).fit_transform(self.data)
        fill_nan_values_df = FillNa(feature, 4).fit_transform(dataset_2_different_dtypes)
        self.assertEqual(dataset_2_different_dtypes.dataframe.fillna(4), fill_nan_values_df.dataframe)
        print('Different dtypes with FillNa succeeded')

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
        print('MinMaxScaler test succeeded')

    def test_new_column_insertion_series_input(self):
        columns_before = len(self.data.columns)
        column = pd.Series([1, 2, 3, 4, 5])
        update_df = Update(column).fit_transform(self.data)
        columns_after = len(update_df.dataframe.columns)
        self.assertEqual(columns_after, columns_before + 1)
        expected_output = pd.concat([self.data, column], axis=1)
        self.assertEqual(expected_output, update_df.dataframe)
        print('Update: insertion of a new column test succeeded')

    def test_update_column_with_pd_series(self):
        feature = 'rating'
        column = pd.Series([1, 2, 3, 4, 5], name=feature)
        update_df = Update(column).fit_transform(self.data)
        self.data[column.name] = column
        self.assertEqual(self.data, update_df.dataframe)
        print('Update: update of an existing column with a pandas Series test succeeded')

    def test_update_multiple_columns_with_pd_df(self):
        columns_dict = {'rating': [1.0, 2.0, 3.0, 4.0, 5.0],
                        'userId': [50, 35, 45, 78, 67]}
        columns_df = pd.DataFrame(columns_dict)
        update_df = Update(columns_df).fit_transform(self.data)
        for col in columns_df.columns:
            self.data[col] = columns_df[col]
        self.assertEqual(self.data, update_df.dataframe)
        print('Update: update of multiple existing columns with a dataframe test succeeded')

    def test_update_multiple_columns_with_dict(self):
        columns_dict = {'rating': [1.0, 2.0, 3.0, 4.0, 5.0],
                        'userId': [50, 35, 45, 78, 67]}
        update_df = Update(columns_dict).fit_transform(self.data)
        for col, val in columns_dict.items():
            self.data[col] = val
        self.assertEqual(self.data, update_df.dataframe)
        print('Update: update of multiple existing columns with a dictionary test succeeded')

    def test_correct_dropna_output(self):
        col_name = 'SomeNaNs'
        column = pd.Series([1, 2, 3], name=col_name)
        update_df = Update(column).fit_transform(self.data)
        drop_nans_df = DropNa(col_name).fit_transform(update_df)
        check_for_nan = drop_nans_df.dataframe[col_name].isnull().values.any()
        self.assertFalse(check_for_nan)
        self.assertEqual(update_df.dataframe.dropna(subset=col_name), drop_nans_df.dataframe)
        print('DropNa test succeeded')

    def test_check_drop_nan_all_df(self):
        col_name = 'SomeNaNs'
        column = pd.Series([1, 2], name=col_name)
        update_df = Update(column).fit_transform(self.data)
        drop_nans_df = DropNa().fit_transform(update_df)
        check_for_nan = drop_nans_df.dataframe.isnull().values.any()
        self.assertFalse(check_for_nan)
        self.assertEqual(update_df.dataframe.dropna(), drop_nans_df.dataframe)
        print('DropNa on all dataframe test succeeded')

    # TODO correct BIN test
    def test_bin_correct_output(self):
        columns_dict = {'rating': [1.0, 2.0, 3.0, 4.0, 5.0]}
        bin_preprocess = PreprocessPipeline([Update(columns_dict),
                                             Bin('rating', 5)])
        new_df = bin_preprocess.fit_transform(self.data)
        expected_dict = {'userId': [1, 1, 1, 1, 1],
                         'movieId': [1, 3, 6, 47, 50],
                         'rating': [1, 2, 3, 4, 5],
                         'timestamp': [964982703, 964981247, 964982224, 964983815, 964982931]}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, new_df.dataframe)
        print('Bin test succeeded')

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
        one_hot_encoding_preprocess = PreprocessPipeline([OneHotEncode('genres', divider='|')])
        cut_feature_df = self.features.head()
        new_df = one_hot_encoding_preprocess.fit_transform(cut_feature_df)
        expected_output = pd.concat(
            [cut_feature_df.drop('genres', axis=1), cut_feature_df['genres'].str.get_dummies('|')],
            axis=1)
        self.assertEqual(new_df.dataframe, expected_output)
        print(new_df)

    # TODO check condense set
    def test_condense_correct_output(self):
        separator = '|'
        condense_preprocess = PreprocessPipeline([Select(['movieId', 'tag']),
                                                  Condense('movieId', separator)])
        new_df = condense_preprocess.fit_transform(self.tags)
        new_df = new_df.dataframe.head()
        expected_dict = {'movieId': [1, 2, 3, 5, 7],
                         'tag': ['pixar|fun', 'fantasy|magic board game|Robin Williams|game',
                                 'moldy|old', 'pregnancy|remake', 'remake']}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, new_df)
        print('Condense test succeeded')

    def test_condense_with_numbers(self):
        test_dataset_dict = {'Id': [1, 1, 2, 1, 4, 5, 4, 2, 5, 4],
                             'tags': [1, 'cart', 2, 'hola', 'ciao', 'uber',
                                      'cab', 3.0, 'NaN', 'Rob']}
        test_dataset = pd.DataFrame(test_dataset_dict)
        condense_df = Condense('Id', '|').fit_transform(test_dataset)
        expected_dict = {'Id': [1, 2, 4, 5],
                         'tags': ['1|cart|hola', '2|3.0',
                                  'ciao|cab|Rob', 'uber|NaN']}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, condense_df.dataframe)
        print('Condense with float and integers test succeeded')

    def test_condense_with_nan(self):
        test_dataset_dict = {'Id': [1, 1, 2, 1, 4, 5, 4, 2, 5, 4],
                             'tags': [1, np.nan, 2, 'hola', 'ciao', 'uber',
                                      'cab', 3.0, 'NaN', np.nan]}
        test_dataset = pd.DataFrame(test_dataset_dict)
        condense_df = Condense('Id', '|').fit_transform(test_dataset)
        expected_dict = {'Id': [1, 2, 4, 5],
                         'tags': ['1|nan|hola', '2|3.0',
                                  'ciao|cab|nan', 'uber|NaN']}
        expected_output = pd.DataFrame(expected_dict)
        self.assertEqual(expected_output, condense_df.dataframe)
        print('Condense with NaNs test succeeded')

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
