from __future__ import annotations
from abc import abstractmethod  # for PreprocessFunction
from itertools import chain  # for threshold binning
from logging import exception
from typing import Any

from scipy.stats._binned_statistic import BinnedStatisticResult

from pandas.api.types import (is_integer_dtype, is_float_dtype)

import numpy as np
import pandas as pd
from lightfm.data import Dataset  # for ToSparseMatrix
from scipy.sparse import coo_matrix  # for matrix inversion
from scipy.stats import binned_statistic  # for BinFeature
from sklearn import preprocessing  # for Normalizer, MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder  # for matrix inversion

from rex import tools


class PreprocessFunction(TransformerMixin, BaseEstimator):
    def apply(self, dataframe):
        if isinstance(dataframe, pd.Series):
            dataframe = pd.DataFrame(dataframe)
        result = self._apply_to_dataframe(pd.DataFrame(dataframe))
        return pd.DataFrame(result) if isinstance(result, pd.Series) else result

    @abstractmethod
    def _apply_to_dataframe(self, dataframe):
        pass


class PreprocessFunctionOnCopy(PreprocessFunction):
    def _apply_to_dataframe(self, dataframe):
        return self._apply_to_copy(dataframe.copy(deep=True))

    @abstractmethod
    def _apply_to_copy(self, dataframe):
        pass


class GroupByFunction(PreprocessFunctionOnCopy):
    def __init__(self, group_by_features: [str] | str | None = None, features: [str] | str | None = None):
        self._group_by_features = list(np.atleast_1d(group_by_features)) if group_by_features is not None else None
        self._features = list(np.atleast_1d(features)) if features is not None else None

    def _apply_to_copy(self, dataframe: pd.DataFrame):
        if self._group_by_features is None and self._features is None:
            for column in dataframe.columns:
                dataframe[[column]] = self._transform(dataframe[[column]])

        elif self._group_by_features is None and self._features is not None:
            dataframe[self._features] = self._transform(dataframe[self._features])

        elif self._group_by_features is not None and self._features is None:
            condition = list(dataframe.columns.difference(self._group_by_features))
            groups = dataframe.groupby(self._group_by_features)[condition]
            dataframe[condition] = groups.apply(lambda x: pd.DataFrame(self._transform(x), index=x.index))

        else:
            groups = dataframe.groupby(self._group_by_features)[self._features]
            dataframe[self._features] = groups.apply(lambda x: pd.DataFrame(self._transform(x), index=x.index))

        return dataframe

    def _transform(self, values: pd.DataFrame) -> pd.DataFrame:
        return self._transform_function(np.atleast_2d(values))

    @abstractmethod
    def _transform_function(self, values: pd.DataFrame) -> pd.DataFrame:
        pass


# --------------UTILITY FUNCTION--------------
def take_cumulative(items, threshold, metric, map_result=lambda x: x):
    remaining_items = items
    cumulate_items = []

    while remaining_items != [] and np.sum(list(map(metric, cumulate_items))) < threshold:
        head, *remaining_items = remaining_items
        cumulate_items.append(head)
    return list(map(map_result, cumulate_items)), list(map(map_result, remaining_items))


def map_series(series, to_exclude, to_replace):
    return series.map(lambda x: x if x in to_exclude else to_replace)


# --------------COO MATRIX--------------
class Select(PreprocessFunctionOnCopy):
    def __init__(self, features):
        self._features = features

    def _apply_to_copy(self, dataframe):
        return dataframe[self._features]


class FillNa(PreprocessFunctionOnCopy):
    def __init__(self, feature, value=None, method=None):
        self._feature = feature
        self._value = value
        self._method = method

    def _apply_to_copy(self, dataframe):
        dataframe[self._feature] = dataframe[self._feature].fillna(value=self._value, method=self._method)
        return dataframe


class Filter(PreprocessFunctionOnCopy):
    def __init__(self, filter_function):
        self._filter_function = filter_function

    def _apply_to_copy(self, dataframe):
        return dataframe[self._filter_function(dataframe)]


class Drop(PreprocessFunctionOnCopy):
    def __init__(self, features):
        self._features = features

    def _apply_to_copy(self, dataframe):
        return dataframe.drop(columns=self._features)


class Update(PreprocessFunctionOnCopy):
    def __init__(self, update):
        self._update = update

    def _apply_to_copy(self, dataframe):
        if isinstance(self._update, pd.Series):
            if self._update.name is not None:
                dataframe[self._update.name] = self._update
            else:
                dataframe = pd.concat([dataframe, self._update], axis=1)
        elif isinstance(self._update, pd.DataFrame):
            for column in self._update.columns:
                dataframe[column] = self._update[column]
        elif isinstance(self._update, dict):
            for column, values in self._update.items():
                dataframe[column] = values
        else:
            raise ValueError('update must be one of Series, DataFrame or a dict')
        return dataframe


# TODO: aggiungere binning complesso
class Bin(PreprocessFunctionOnCopy):
    def __init__(self, feature, bins, baseline=0):
        self._feature = feature
        self._bins = bins
        self._baseline = baseline

    def _apply_to_copy(self, dataframe):
        binned_statistic_result: BinnedStatisticResult = binned_statistic(
            dataframe[self._feature],
            dataframe[self._feature],
            bins=self._bins)

        dataframe[self._feature] = binned_statistic_result.binnumber + self._baseline

        return dataframe


class BinThreshold(PreprocessFunctionOnCopy):
    def __init__(self, feature, binning_value, threshold=10, divider=None):
        self._feature = feature
        self._binning_value = binning_value
        self._threshold = threshold
        self._divider = divider

    def _apply_to_copy(self, dataframe):
        if self._divider is None:
            column = dataframe[self._feature]
            to_exclude = self._calculate_to_exclude(column)
            dataframe[self._feature] = map_series(column, to_exclude, self._binning_value)
        else:
            items_features = dataframe[self._feature].str.split(self._divider).values
            all_features = chain(*items_features)
            to_exclude = self._calculate_to_exclude(pd.Series(all_features))
            dataframe[self._feature] = [
                self._divider.join(
                    {feature if feature in to_exclude else self._binning_value for feature in item_features})
                for item_features in items_features]

        return dataframe

    def _calculate_to_exclude(self, values):
        value_counts_percent = values.value_counts(normalize=True) * 100
        return value_counts_percent[value_counts_percent.values > self._threshold].index


class BinCumulative(PreprocessFunctionOnCopy):
    def __init__(self, feature, binning_value, cumulative_threshold=70, divider=None):
        self._feature = feature
        self._binning_value = binning_value
        self._cumulative_threshold = cumulative_threshold
        self._divider = divider

    def _apply_to_copy(self, dataframe):
        if self._divider is None:
            column = dataframe[self._feature]
            to_exclude = self._calculate_to_exclude(column)
            dataframe[self._feature] = map_series(column, to_exclude, self._binning_value)
        else:
            items_features = dataframe[self._feature].str.split(self._divider).values
            all_features = chain(*items_features)
            to_exclude = self._calculate_to_exclude(pd.Series(all_features))
            dataframe[self._feature] = [
                self._divider.join(
                    {feature if feature in to_exclude else self._binning_value for feature in item_features})
                for item_features in items_features]
        return dataframe

    def _calculate_to_exclude(self, values):
        value_counts_percent = values.value_counts(normalize=True) * 100
        to_exclude, _ = take_cumulative(list(value_counts_percent.items()),
                                        self._cumulative_threshold, lambda x: x[1],
                                        lambda x: x[0])
        return to_exclude


class DropNa(PreprocessFunction):
    def __init__(self, subset_features=None):
        self._subset_features = subset_features

    def _apply_to_dataframe(self, dataframe):
        return dataframe.dropna(subset=self._subset_features)


class DropDuplicates(PreprocessFunctionOnCopy):
    def __init__(self, subset_features=None, keep="last"):
        self._subset_features = subset_features
        self._keep = keep

    def _apply_to_copy(self, dataframe):
        return dataframe.drop_duplicates(subset=self._subset_features, keep=self._keep)


class Clip(PreprocessFunctionOnCopy):
    def __init__(self, feature, lower, upper):
        self._feature = feature
        self._lower = lower
        self._upper = upper

    def _apply_to_copy(self, dataframe):
        dataframe[self._feature] = dataframe[self._feature].clip(lower=self._lower, upper=self._upper)
        return dataframe


class Map(PreprocessFunctionOnCopy):
    def __init__(self, feature, arg):
        self._feature = feature
        self._arg = arg

    def _apply_to_copy(self, dataframe):
        def mapping_function(value):
            if isinstance(self._arg, dict):
                return value if value not in self._arg else self._arg[value]
            else:
                return self._arg(value)

        dataframe[self._feature] = dataframe[self._feature].map(mapping_function)
        return dataframe


class OneHotEncode(PreprocessFunction):
    def __init__(self, features, divider=None):
        self._features = np.atleast_1d(features)
        self._divider = divider

    def _apply_to_dataframe(self, dataframe):
        for feature in self._features:
            one_hot_feature_matrix = pd.get_dummies(dataframe[feature]) if self._divider is None \
                else dataframe[feature].str.get_dummies(self._divider)
            dataframe = pd.concat([dataframe.drop(feature, axis=1), one_hot_feature_matrix], axis=1)

        return dataframe


class ToDenseMatrix(PreprocessFunction):
    def __init__(self, user_column, item_column):
        self._user_column = user_column
        self._item_column = item_column

    def _apply_to_dataframe(self, dataframe):
        matrix_generator = Dataset()
        user_ids = np.unique(dataframe[self._user_column])
        item_ids = np.unique(dataframe[self._item_column])

        matrix_generator.fit(
            users=user_ids,
            items=item_ids,
        )
        _, weights = matrix_generator.build_interactions(dataframe.values)
        weights_dataframe = pd.DataFrame(data=weights.todense(), index=user_ids, columns=item_ids)
        return weights_dataframe


class StandardScaler(GroupByFunction):
    def _transform_function(self, values: pd.DataFrame) -> pd.DataFrame:
        return preprocessing.StandardScaler().fit_transform(values)


class MinMaxScaler(GroupByFunction):
    def _transform_function(self, values: pd.DataFrame) -> pd.DataFrame:
        return preprocessing.MinMaxScaler().fit_transform(values)


class Normalizer(GroupByFunction):
    def __init__(self,
                 group_by_features: [str] | str | None = None,
                 features: [str] | str | None = None,
                 norm: str = 'l2'):
        super(Normalizer, self).__init__(group_by_features=group_by_features, features=features)
        self._norm = norm

    def _transform_function(self, values: pd.DataFrame) -> pd.DataFrame:
        return preprocessing.Normalizer(norm=self._norm).fit_transform(values.T).T


class Condense(PreprocessFunction):
    def __init__(self, feature, join_separator):
        self._feature = feature
        self._join_separator = join_separator

    def _apply_to_dataframe(self, dataframe: pd.DataFrame):
        return dataframe.groupby(self._feature).agg(self._aggregate).reset_index()

    def _aggregate(self, group):
        return self._join_separator.join(group.astype(str).unique())


class ToCOOMatrix(PreprocessFunction):
    def __init__(self, user_column_name, item_column_name, weights_column_name):
        self._user_column_name = user_column_name
        self._item_column_name = item_column_name
        self._weights_column_name = weights_column_name

    def _apply_to_dataframe(self, utility_matrix):
        user_encoder = LabelEncoder().fit(list(utility_matrix.index))
        item_encoder = LabelEncoder().fit(list(utility_matrix.columns))
        coo = coo_matrix(utility_matrix.values)
        return pd.DataFrame({
            self._user_column_name: user_encoder.inverse_transform(coo.row),
            self._item_column_name: item_encoder.inverse_transform(coo.col),
            self._weights_column_name: coo.data})


# --------------SPARSE MATRIX--------------
"""
class MinMaxScalerValues(PreprocessFunction):
    def _apply_to_dataframe(self, dataframe):
        min_max_scaled_data = preprocessing.MinMaxScaler().fit_transform(dataframe.T).T
        return pd.DataFrame(min_max_scaled_data, columns=dataframe.columns, index=dataframe.index)


class StandardScaler(PreprocessFunction):
    def _apply_to_dataframe(self, dataframe):
        scaled_data = preprocessing.StandardScaler().fit_transform(dataframe.T).T
        return pd.DataFrame(scaled_data, columns=dataframe.columns, index=dataframe.index)


class Normalizer(PreprocessFunction):
    def __init__(self, norm):
        self._norm = norm

    def _apply_to_dataframe(self, dataframe):
        normalized_data = preprocessing.Normalizer(self._norm).fit_transform(dataframe)
        return pd.DataFrame(normalized_data, columns=dataframe.columns, index=dataframe.index)


class BinValues(PreprocessFunction):
    def __init__(self, bins, baseline=0):
        self._bins = bins
        self._baseline = baseline

    def apply(self, dataframe):
        values = dataframe.values.flatten()
        binned_values = binned_statistic(
            values,
            values,
            bins=self._bins).binnumber + self._baseline
        return pd.DataFrame(binned_values.reshape(dataframe.shape), columns=dataframe.columns, index=dataframe.index)


class MapValues(PreprocessFunction):
    def __init__(self, mapping_values):
        self._mapping_values = mapping_values

    def _apply_to_copy(self, dataframe):
        mapping_function = np.vectorize(lambda x: x if x not in self._mapping_values else self._mapping_values[x])
        return pd.DataFrame(mapping_function(dataframe.values), columns=dataframe.columns, index=dataframe.index)

"""


# -----------PIPELINE---------------


class PreprocessPipeline(PreprocessFunction):
    def __init__(self, preprocess_functions, verbose=1, force=False):
        self._preprocess_functions = preprocess_functions
        self._verbose = verbose
        self._force = force

    def _apply_to_dataframe(self, dataframe):
        if self._verbose > 1:
            print(dataframe)
        return self._apply_recursive(dataframe, self._preprocess_functions)

    def _apply_recursive(self, dataframe, functions):
        if not functions:
            return dataframe
        else:
            current_function, *remaining_functions = functions
            function_name = type(current_function).__name__
            if self._verbose > 0:
                print(f"Starting {function_name}")
            try:
                applied_function = current_function.apply(dataframe)
                if self._verbose > 1:
                    print(applied_function)
                if self._verbose > 0:
                    print(f"{function_name} successfully")
                    print()
                return self._apply_recursive(applied_function, remaining_functions)
            except:
                exception(f'{function_name} exception')
                if self._force:
                    return self._apply_recursive(dataframe, remaining_functions)
                else:
                    return dataframe


class DatasetsPreprocessor:
    def __init__(self, dataset_pipeline=None, user_features_pipeline=None, item_features_pipeline=None):
        self._dataset_pipeline = dataset_pipeline
        self._user_feature_pipeline = user_features_pipeline
        self._item_feature_pipeline = item_features_pipeline

    def apply(self, dataset=None, user_features=None, item_features=None):

        def apply_maybe_pipeline(dataframe, pipeline):
            if pipeline is None and dataframe is not None:
                return dataframe
            elif dataframe is None:
                return None
            else:
                return pipeline.transform(dataframe)

        new_weights_dataframe = apply_maybe_pipeline(dataset, self._dataset_pipeline)
        new_user_features_dataframe = apply_maybe_pipeline(user_features, self._user_feature_pipeline)
        new_item_features_dataframe = apply_maybe_pipeline(item_features, self._item_feature_pipeline)
        return new_weights_dataframe, new_user_features_dataframe, new_item_features_dataframe


# -------------AUTO PREPROCESS-------------
def _base_transformation(dataframe, duplicates_subset, verbose=True):
    if any(dataframe[column].isna().sum() > 0 for column in dataframe.columns):
        if verbose:
            print('Applied DropNa transformation')
        dataframe = DropNa().apply(dataframe)

    if dataframe.duplicated(subset=duplicates_subset).sum() > 0:
        if verbose:
            print(f'Applied DropDuplicates transformation to {duplicates_subset} keeping last value')
        dataframe = DropDuplicates().apply(dataframe)

    return dataframe


def auto_preprocess_weights_dataframe(dataframe, verbose=True):
    tools.check_is_dataframe(dataframe)

    if dataframe.columns.size < 2:
        raise ValueError('DataFrame must have at least the column with user ids and a column with item ids')

    if dataframe.columns.size > 3:
        if verbose:
            print(f'Selected only the first 3 columns')
        dataframe = dataframe.iloc[:, :3].copy(deep=True)

    dataframe = _base_transformation(dataframe, dataframe.columns[:2], verbose=verbose)

    if dataframe.columns.size == 2:
        return dataframe

    weights = dataframe.iloc[:, 2]
    weights_column = dataframe.columns[2]

    if not is_float_dtype(weights.dtype) and not is_integer_dtype(weights.dtype):
        unique_values = np.unique(weights.values)
        mapping_values = dict(zip(unique_values, preprocessing.LabelEncoder().fit_transform(unique_values)))
        if verbose:
            print(f'Applied Map transformation to feature \'{weights_column}\' with values {mapping_values}')
        dataframe = Map(weights_column, mapping_values).apply(dataframe)

    weights = dataframe.iloc[:, 2]
    if weights.max() > 1 or weights.min() < 0:
        if verbose:
            print(f'Applied MinMaxScaler transformation to feature \'{weights_column}\'')
        dataframe = MinMaxScaler(features=weights_column).apply(dataframe)

    return dataframe


def auto_preprocess_features_dataframe(dataframe: pd.DataFrame, verbose=True) -> pd.DataFrame:
    tools.check_is_dataframe(dataframe)

    if dataframe.columns.size < 2:
        raise ValueError("Features DataFrame must have id column and at least one feature column")

    dataframe = _base_transformation(dataframe, dataframe.columns[0], verbose=verbose)

    for feature in dataframe.iloc[:, 1:].columns:
        feature_col = dataframe[feature]
        if (is_float_dtype(feature_col.dtype) or is_integer_dtype(feature_col.dtype)) and \
                (feature_col.min() < 0 or feature_col.max() > 1):
            if verbose:
                print(f'Applied MinMaxScaler transformation to feature \'{feature}\'')
            dataframe = MinMaxScaler(features=feature).apply(dataframe)

        if not is_float_dtype(feature_col.dtype) and not is_integer_dtype(feature_col.dtype):
            if feature_col.str.contains("|", regex=False).sum() > 0:
                if verbose:
                    print(f'''Applied BinCumulative at 70% to feature \'{feature}\' 
                    with new value \'other\' 
                    and using | as divider''')
                dataframe = BinCumulative(feature, 'other', divider='|').apply(dataframe)
                if verbose:
                    print(f'Applied OneHotEncode transformation to feature \'{feature}\' using | as divider')
                dataframe = OneHotEncode(features=feature, divider='|').apply(dataframe)
            else:
                if verbose:
                    print(f'Applied BinCumulative at 70% to feature \'{feature}\'')
                dataframe = BinCumulative(feature, 'other').apply(dataframe)
    return dataframe
