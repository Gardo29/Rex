from __future__ import annotations
from abc import abstractmethod, ABC  # for PreprocessFunction
from itertools import chain  # for threshold binning
from logging import exception
from typing import Any, Iterable, Union

from numpy.random import RandomState
from scipy.stats._binned_statistic import BinnedStatisticResult
from sklearn.utils.validation import check_is_fitted

from rex import tools
from pandas.api.types import (is_integer_dtype, is_float_dtype)

import numpy as np
from pandas import DataFrame
import pandas as pd
from lightfm.data import Dataset  # for ToSparseMatrix
from scipy.sparse import coo_matrix  # for matrix inversion
from scipy.stats import binned_statistic  # for BinFeature
from sklearn import preprocessing  # for Normalizer, MinMaxScaler, StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder  # for matrix inversion

from rex import tools
from rex.model_selection import train_test_split


class PreprocessedDataFrame:
    def __init__(self, dataframe: DataFrame, preprocess_functions: list[PreprocessFunction]):
        if dataframe is None or not isinstance(dataframe, DataFrame):
            raise ValueError("'dataframe' must be a Pandas DataFrame")
        if preprocess_functions is None or \
                preprocess_functions == [] or \
                not all(isinstance(function, PreprocessFunction) for function in preprocess_functions):
            raise ValueError("'preprocess_pipeline' must be a non empty list of PreprocessFunction")
        self._dataframe = dataframe
        self._preprocess_pipeline = preprocess_functions.copy()

    def __str__(self):
        return f"""{self.dataframe}\n {self.preprocess_functions}"""

    def __eq__(self, other):
        if isinstance(other, PreprocessedDataFrame):
            return other.dataframe.equals(self._dataframe) and \
                   np.array_equal(other.preprocess_functions, self._preprocess_pipeline)
        return False

    @property
    def preprocess_functions(self) -> list[PreprocessFunction]:
        return self._preprocess_pipeline.copy()

    @property
    def dataframe(self):
        return self._dataframe.copy(deep=True)


class PreprocessFunction(TransformerMixin, BaseEstimator, ABC):
    def fit(self, input_data: DataFrame | PreprocessedDataFrame, y=None) -> PreprocessFunction:
        if isinstance(input_data, pd.Series):
            return self._fit(DataFrame(input_data))
        elif isinstance(input_data, PreprocessedDataFrame):
            return self._fit(input_data.dataframe)
        elif isinstance(input_data, DataFrame):
            return self._fit(input_data)
        else:
            raise ValueError("'input' must be a DataFrame or PreprocessedDataFrame")

    def transform(self, input_data: DataFrame | PreprocessedDataFrame, y=None) -> PreprocessedDataFrame:
        if isinstance(input_data, pd.Series):
            dataframe = DataFrame(input_data)
            preprocess_functions = []
        elif isinstance(input_data, PreprocessedDataFrame):
            dataframe = input_data.dataframe
            preprocess_functions = input_data.preprocess_functions
        elif isinstance(input_data, DataFrame):
            dataframe = input_data
            preprocess_functions = []
        else:
            raise ValueError("'input' must be a DataFrame or PreprocessedDataFrame")

        preprocess_functions.append(self)
        result = self._transform(dataframe)
        dataframe = DataFrame(result) if isinstance(result, pd.Series) else result
        return PreprocessedDataFrame(dataframe, preprocess_functions)

    def fit_transform(self,
                      dataframe: DataFrame | PreprocessedDataFrame,
                      y=None,
                      **fit_params) -> PreprocessedDataFrame:
        self.fit(dataframe)
        return self.transform(dataframe)

    @abstractmethod
    def _transform(self, dataframe: DataFrame) -> pd.Series | DataFrame:
        pass

    def _fit(self, dataframe: DataFrame | PreprocessedDataFrame) -> PreprocessFunction:
        return self

    def __eq__(self, other):
        return isinstance(other, self.__class__)


class PreprocessFunctionOnCopy(PreprocessFunction, ABC):
    def _transform(self, dataframe):
        return self._apply_to_copy(dataframe.copy(deep=True))

    @abstractmethod
    def _apply_to_copy(self, dataframe):
        pass


# --------------UTILITY FUNCTION--------------
def take_cumulative(items, threshold, metric, map_result=lambda x: x):
    remaining_items = items
    cumulate_items = []

    while remaining_items != [] and np.sum(list(map(metric, cumulate_items))) < threshold:
        head, *remaining_items = remaining_items
        cumulate_items.append(head)
    return set(map(map_result, cumulate_items)), set(map(map_result, remaining_items))


def keep_or_replace(value, to_keep, replace_value):
    return value if value in to_keep else replace_value


# --------------COO MATRIX--------------
class Select(PreprocessFunctionOnCopy):
    def __init__(self, features):
        self.features = features

    def _apply_to_copy(self, dataframe):
        return dataframe[self.features]

    def __eq__(self, other):
        return super().__eq__(other) and np.array_equal(self.features, other.features)


class FillNa(PreprocessFunctionOnCopy):
    def __init__(self, feature, value=None, method=None):
        self.feature = feature
        self.value = value
        self.method = method

    def _apply_to_copy(self, dataframe):
        dataframe[self.feature] = dataframe[self.feature].fillna(value=self.value, method=self.method)
        return dataframe

    def __eq__(self, other):
        return super().__eq__(
            other) and self.feature == other.feature and self.value == other.value and self.method == self.method


class Filter(PreprocessFunctionOnCopy):
    def __init__(self, filter_function):
        self.filter_function = filter_function

    def _apply_to_copy(self, dataframe):
        return dataframe[self.filter_function(dataframe)]

    def __eq__(self, other):
        return super().__eq__(other) and self.filter_function == other.filter_function


class Drop(PreprocessFunctionOnCopy):
    def __init__(self, features):
        self.features = features

    def _apply_to_copy(self, dataframe):
        return dataframe.drop(columns=self.features)

    def __eq__(self, other):
        return super().__eq__(other) and np.array_equal(self.features, other.features)


class Rename(PreprocessFunctionOnCopy):
    def __init__(self, columns_dict: dict[str, str]):
        self.columns_dict = columns_dict

    def _apply_to_copy(self, dataframe):
        return dataframe.rename(columns=self.columns_dict)

    def __eq__(self, other):
        return super().__eq__(other) and self.columns_dict == other.columns_dict


class Update(PreprocessFunctionOnCopy):
    def __init__(self, update):
        self.update = update

    def _apply_to_copy(self, dataframe):
        if isinstance(self.update, pd.Series):
            if self.update.name is not None:
                dataframe[self.update.name] = self.update
            else:
                dataframe = pd.concat([dataframe, self.update], axis=1)
        elif isinstance(self.update, DataFrame):
            for column in self.update.columns:
                dataframe[column] = self.update[column]
        elif isinstance(self.update, dict):
            for column, values in self.update.items():
                dataframe[column] = values
        else:
            raise ValueError('update must be one of Series, DataFrame or a dict or a number')
        return dataframe

    def __eq__(self, other):
        return super().__eq__(other) and self.update == other.update


# TODO: aggiungere binning complesso
class Bin(PreprocessFunctionOnCopy):
    def __init__(self, feature, bins, baseline=0):
        self.feature = feature
        self.bins = bins
        self.baseline = baseline

    def _fit(self, dataframe: DataFrame, y=None) -> Bin:
        pre_binning: BinnedStatisticResult = binned_statistic(
            dataframe[self.feature],
            dataframe[self.feature],
            bins=self.bins)
        self.bin_edges_ = pre_binning.bin_edges
        return self

    def _apply_to_copy(self, dataframe):
        check_is_fitted(self)
        binned_statistic_result: BinnedStatisticResult = binned_statistic(
            dataframe[self.feature],
            dataframe[self.feature],
            bins=self.bin_edges_)

        dataframe[self.feature] = binned_statistic_result.binnumber + self.baseline

        return dataframe

    def __eq__(self, other):
        return super().__eq__(
            other) and self.feature == other.feature and self.bins == other.bins and self.baseline == other.baseline


class BinDensity(PreprocessFunctionOnCopy, ABC):
    def __init__(self, feature, binning_value, divider=None):
        self.feature = feature
        self.binning_value = binning_value
        self.divider = divider

    def _fit(self, dataframe: DataFrame, y=None) -> BinDensity:
        if self.divider is None:
            column = dataframe[self.feature]
        else:
            items_features = dataframe[self.feature].str.split(self.divider).values
            all_features = chain(*items_features)
            column = pd.Series(all_features)

        self.features_ = set(column.values)
        self.excluted_features_, self.grouped_features_ = self._split(column)
        return self

    def _apply_to_copy(self, dataframe):
        check_is_fitted(self)
        column = dataframe[self.feature]

        values = column.values if self.divider is None else column.str.split(self.divider).values
        extra_features = set(values) - set(self.features_)
        # error check
        if extra_features:
            raise ValueError(f'{extra_features} no present during fit call')
        # transformation
        if self.divider is None:
            dataframe[self.feature] = column.map(
                lambda x: keep_or_replace(x, self.excluted_features_, self.binning_value))
        else:
            dataframe[self.feature] = [
                self.divider.join(
                    {keep_or_replace(feature, self.excluted_features_, self.binning_value)
                     for feature in multiple_feature})
                for multiple_feature in values]

        return dataframe

    @abstractmethod
    def _split(self, column: pd.Series) -> (set[Any], set[Any]):
        pass

    def __eq__(self, other):
        return super().__eq__(other) and \
               self.feature == other.feature and \
               self.binning_value == other.binning_value and \
               self.divider == other.divider


class BinThreshold(BinDensity):
    def __init__(self, feature, binning_value, threshold=10, divider=None):
        super(BinThreshold, self).__init__(feature, binning_value, divider)
        self.threshold = threshold

    def _split(self, values):
        value_counts_percent = values.value_counts(normalize=True) * 100
        select_over_threshold = value_counts_percent.values >= self.threshold
        return set(value_counts_percent[select_over_threshold].index), set(
            value_counts_percent[~select_over_threshold].index)

    def __eq__(self, other):
        return super().__eq__(other) and self.threshold == other.threshold


class BinCumulative(BinThreshold):
    def __init__(self, feature, binning_value, cumulative_threshold=70, divider=None):
        super(BinCumulative, self).__init__(feature, binning_value, divider)
        self.cumulative_threshold = cumulative_threshold

    def _split(self, values):
        value_counts_percent = values.value_counts(normalize=True) * 100
        return take_cumulative(list(value_counts_percent.items()),
                               self.cumulative_threshold, lambda x: x[1],
                               lambda x: x[0])

    def __eq__(self, other):
        return super().__eq__(other) and self.cumulative_threshold == other.cumulative_threshold


class DropNa(PreprocessFunction):
    def __init__(self, subset_features=None):
        self.subset_features = subset_features

    def _transform(self, dataframe):
        return dataframe.dropna(subset=self.subset_features)

    def __eq__(self, other):
        return super().__eq__(other) and np.array_equal(self.subset_features, other.subset_features)


class DropDuplicates(PreprocessFunctionOnCopy):
    def __init__(self, subset_features=None, keep="last"):
        self.subset_features = subset_features
        self.keep = keep

    def _apply_to_copy(self, dataframe):
        return dataframe.drop_duplicates(subset=self.subset_features, keep=self.keep)

    def __eq__(self, other):
        return super().__eq__(other) and \
               np.array_equal(self.subset_features, other.subset_features) and \
               self.keep == other.keep


class Clip(PreprocessFunctionOnCopy):
    def __init__(self, feature, lower, upper):
        self.feature = feature
        self.lower = lower
        self.upper = upper

    def _apply_to_copy(self, dataframe):
        dataframe[self.feature] = dataframe[self.feature].clip(lower=self.lower, upper=self.upper)
        return dataframe

    def __eq__(self, other):
        return super().__eq__(other) and \
               self.feature == other.feature and \
               self.lower == other.lower and \
               self.upper == other.upper


class Map(PreprocessFunctionOnCopy):
    def __init__(self, feature, arg=None):
        self.feature = feature
        self.arg = arg

    def _apply_to_copy(self, dataframe):
        default_baseline = 1
        feature = dataframe[self.feature]
        if self.arg is None:
            unique_values = tools.unique(feature.values)
            mapping_values = dict(
                zip(unique_values, preprocessing.LabelEncoder().fit_transform(unique_values) + default_baseline))
            dataframe[self.feature] = feature.map(mapping_values)
        elif isinstance(self.arg, dict):
            dataframe[self.feature] = feature.map(lambda x: x if x not in self.arg else self.arg[x])
        else:
            dataframe[self.feature] = feature.map(self.arg)

        return dataframe

    def __eq__(self, other):
        return super().__eq__(other) and self.feature == other.feature and self.arg == other.arg


class OneHotEncode(PreprocessFunction):
    def __init__(self, features, divider=None):
        self.features = list(np.atleast_1d(features))
        self.divider = divider

    def _transform(self, dataframe):
        for feature in self.features:
            one_hot_feature_matrix = pd.get_dummies(dataframe[feature]) if self.divider is None \
                else dataframe[feature].str.get_dummies(self.divider)
            dataframe = pd.concat([dataframe.drop(feature, axis=1), one_hot_feature_matrix], axis=1)

        return dataframe

    def __eq__(self, other):
        return super().__eq__(other) and \
               np.array_equal(self.features, other.features) and \
               self.divider == other.divider


class GroupByFunction(PreprocessFunctionOnCopy, ABC):
    def __init__(self, group_by_features: [str] | str | None = None, features: [str] | str | None = None):
        self.group_by_features = list(np.atleast_1d(group_by_features)) if group_by_features is not None else None
        self.features = list(np.atleast_1d(features)) if features is not None else None

    def _apply_to_copy(self, dataframe: DataFrame):
        if self.group_by_features is None and self.features is None:
            for column in dataframe:
                dataframe[[column]] = self._grouped_transform(dataframe[[column]])

        elif self.group_by_features is None and self.features is not None:
            dataframe[self.features] = self._grouped_transform(dataframe[self.features])

        elif self.group_by_features is not None and self.features is None:
            condition = list(dataframe.columns.difference(self.group_by_features))
            groups = dataframe.groupby(self.group_by_features)[condition]
            dataframe[condition] = groups.apply(lambda x: DataFrame(self._grouped_transform(x), index=x.index))

        else:
            groups = dataframe.groupby(self.group_by_features)[self.features]
            dataframe[self.features] = groups.apply(lambda x: DataFrame(self._grouped_transform(x), index=x.index))

        return dataframe

    def _grouped_transform(self, values: DataFrame) -> DataFrame:
        return self._transform_function(np.atleast_2d(values))

    @abstractmethod
    def _transform_function(self, values: DataFrame) -> DataFrame:
        pass

    def __eq__(self, other):
        return super().__eq__(other) and \
               np.array_equal(self.features, other.features) and \
               np.array_equal(self.group_by_features, other.group_by_features)


# TODO: come fare?
class StandardScaler(GroupByFunction):
    def _fit(self, dataframe: DataFrame, y=None) -> PreprocessFunction:
        pass

    def _transform_function(self, values: DataFrame) -> DataFrame:
        return preprocessing.StandardScaler().fit_transform(values)


class MinMaxScaler(GroupByFunction):
    def _transform_function(self, values: DataFrame) -> DataFrame:
        return preprocessing.MinMaxScaler().fit_transform(values)


class Normalizer(GroupByFunction):
    def __init__(self,
                 group_by_features: [str] | str | None = None,
                 features: [str] | str | None = None,
                 norm: str = 'l2'):
        super(Normalizer, self).__init__(group_by_features=group_by_features, features=features)
        self.norm = norm

    def _transform_function(self, values: DataFrame) -> DataFrame:
        return preprocessing.Normalizer(norm=self.norm).fit_transform(values.T).T

    def __eq__(self, other):
        return super().__eq__(other) and self.norm == other.norm


class Condense(PreprocessFunction):
    def __init__(self, feature, join_separator):
        self.feature = feature
        self.join_separator = join_separator

    def _transform(self, dataframe: DataFrame):
        return dataframe.groupby(self.feature).agg(self._aggregate).reset_index()

    def _aggregate(self, group):
        return self.join_separator.join(group.astype(str).unique())

    def __eq__(self, other):
        return super().__eq__(other) and self.feature == other.feature and self.join_separator == other.join_separator


class ToCOOMatrix(PreprocessFunction):
    def __init__(self, user_column_name: str, item_column_name: str, weights_column_name: str):
        self.user_column_name = user_column_name
        self.item_column_name = item_column_name
        self.weights_column_name = weights_column_name

    def _transform(self, utility_matrix):
        user_encoder = LabelEncoder().fit(list(utility_matrix.index))
        item_encoder = LabelEncoder().fit(list(utility_matrix.columns))
        coo = coo_matrix(utility_matrix.values)
        return DataFrame({
            self.user_column_name: user_encoder.inverse_transform(coo.row),
            self.item_column_name: item_encoder.inverse_transform(coo.col),
            self.weights_column_name: coo.data})

    def __eq__(self, other):
        return super().__eq__(other) and \
               self.user_column_name == other.user_column_name and \
               self.item_column_name == other.item_column_name and \
               self.weights_column_name == other.weights_column_name


# --------------SPARSE MATRIX--------------
"""
class MinMaxScalerValues(PreprocessFunction):
    def _apply_to_dataframe(self, dataframe):
        min_max_scaled_data = preprocessing.MinMaxScaler().fit_transform(dataframe.T).T
        return DataFrame(min_max_scaled_data, columns=dataframe.columns, index=dataframe.index)


class StandardScaler(PreprocessFunction):
    def _apply_to_dataframe(self, dataframe):
        scaled_data = preprocessing.StandardScaler().fit_transform(dataframe.T).T
        return DataFrame(scaled_data, columns=dataframe.columns, index=dataframe.index)


class Normalizer(PreprocessFunction):
    def __init__(self, norm):
        self._norm = norm

    def _apply_to_dataframe(self, dataframe):
        normalized_data = preprocessing.Normalizer(self._norm).fit_transform(dataframe)
        return DataFrame(normalized_data, columns=dataframe.columns, index=dataframe.index)


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
        return DataFrame(binned_values.reshape(dataframe.shape), columns=dataframe.columns, index=dataframe.index)


class MapValues(PreprocessFunction):
    def __init__(self, mapping_values):
        self._mapping_values = mapping_values

    def _apply_to_copy(self, dataframe):
        mapping_function = np.vectorize(lambda x: x if x not in self._mapping_values else self._mapping_values[x])
        return DataFrame(mapping_function(dataframe.values), columns=dataframe.columns, index=dataframe.index)

"""


# -----------PIPELINE---------------


class PreprocessPipeline(TransformerMixin, BaseEstimator):
    def __init__(self, preprocess_functions: Iterable[PreprocessFunction], verbose=1, force=False):
        if not isinstance(preprocess_functions, Iterable):
            raise ValueError("'preprocess_functions' must be an Iterable of PreprocessFunctions")
        self.preprocess_functions = list(preprocess_functions)
        self.verbose = verbose
        self.force = force

    def fit(self, dataframe: DataFrame, y=None) -> PreprocessPipeline:
        for function in self.preprocess_functions:
            function.fit(dataframe)
        return self

    def transform(self, dataframe) -> PreprocessedDataFrame | DataFrame:
        if self.verbose > 1:
            print(dataframe)
        return self._apply_recursive(dataframe, self.preprocess_functions)

    def set_verbose(self, verbose) -> PreprocessPipeline:
        return PreprocessPipeline(self.preprocess_functions, verbose=verbose)

    def _apply_recursive(self,
                         dataframe: DataFrame | PreprocessedDataFrame,
                         functions: Iterable[PreprocessFunction]):
        if not functions:
            return dataframe
        else:
            current_function, *remaining_functions = functions
            function_name = type(current_function).__name__
            if self.verbose > 0:
                print(f"Starting '{function_name}'")
            try:
                applied_function = current_function.transform(dataframe)
                if self.verbose > 1:
                    print(applied_function)
                if self.verbose > 0:
                    print(f"'{function_name}' successfully")
                    print()
                return self._apply_recursive(applied_function, remaining_functions)
            except Exception:
                exception(f"'{function_name}' exception")
                if self.force:
                    return self._apply_recursive(dataframe, remaining_functions)
                else:
                    return dataframe

    def fit_transform(self, dataframe: DataFrame, y=None, **fit_params):
        self.fit(dataframe)
        return self.transform(dataframe)

    def __eq__(self, other):
        if isinstance(other, PreprocessPipeline):
            return self.preprocess_functions == other.preprocess_functions
        else:
            return False


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


# ------------ AUTO PREPROCESS -----------


def auto_preprocess_weights_dataframe(dataframe: DataFrame,
                                      train_size: int | float | None = None,
                                      sparse_input: bool = False,
                                      random_state: int | RandomState | None = None,
                                      return_pipeline: bool = False,
                                      verbose: bool = True) -> Union[DataFrame,
                                                                     (DataFrame, PreprocessPipeline),
                                                                     (DataFrame, DataFrame, PreprocessPipeline),
                                                                     PreprocessedDataFrame,
                                                                     (PreprocessedDataFrame, PreprocessPipeline),
                                                                     (PreprocessedDataFrame, PreprocessedDataFrame,
                                                                      PreprocessPipeline)]:
    user_id_col = 0
    weights_col = 2
    max_columns = 3
    min_columns = 2

    tools.is_dataframe(dataframe)

    if tools.is_no_weights_dataframe(dataframe):
        if train_size:
            train, test = train_test_split(dataframe, train_size, random_state)
            return (train, test, PreprocessPipeline([], verbose=verbose)) if return_pipeline else (train, test)
        else:
            return dataframe, PreprocessPipeline([], verbose=verbose) if return_pipeline else dataframe

    if sparse_input:
        dataframe = ToCOOMatrix('user_id', 'item_id', 'rating').fit_transform(dataframe)

    if tools.get_df(dataframe).columns.size < min_columns:
        raise ValueError('DataFrame must have at least the column with user ids and a column with item ids')

    if tools.get_df(dataframe).columns.size > max_columns:
        if verbose:
            print(f'Selected only the first 3 columns')
        dataframe = Select(list(tools.get_df(dataframe).columns[:max_columns])).fit_transform(dataframe)

    if tools.has_na(dataframe, verbose):
        dataframe = DropNa().fit_transform(dataframe)

    if tools.has_duplicates(dataframe, tools.get_df(dataframe).columns[:weights_col], verbose):
        dataframe = DropDuplicates(
            subset_features=list(tools.get_df(dataframe).columns[:weights_col])).fit_transform(
            dataframe)

    if tools.is_categorical(tools.get_df(dataframe).iloc[:, weights_col], verbose):
        dataframe = Map(tools.get_df(dataframe).columns[weights_col]).fit_transform(dataframe)

    if tools.is_not_normalized(tools.get_df(dataframe).iloc[:, weights_col], verbose):
        if verbose:
            print(
                f"Applied L2 Normalizer to {tools.get_df(dataframe).columns[weights_col]} column"
                f" using '{tools.get_df(dataframe).columns[user_id_col]}' as group")
        dataframe = Normalizer(group_by_features=tools.get_df(dataframe).columns[user_id_col],
                               features=tools.get_df(dataframe).columns[weights_col]).fit_transform(dataframe)

    # split
    if train_size:
        train, test = train_test_split(dataframe, train_size, random_state)
        # split and is PreprocessedDataFrame
        if isinstance(dataframe, PreprocessedDataFrame):
            if return_pipeline:
                return train, test, PreprocessPipeline(dataframe.preprocess_functions, verbose=verbose)
            return train, test
        # split and is DataFrame
        return (train, test, PreprocessPipeline([], verbose=verbose)) if return_pipeline else (train, test)
    # don't split and PreprocessedDataFrame
    elif isinstance(dataframe, PreprocessedDataFrame):
        if return_pipeline:
            return dataframe, PreprocessPipeline(dataframe.preprocess_functions, verbose=verbose)
        return dataframe
    # don't split and DataFrame
    else:
        return (dataframe, PreprocessPipeline([], verbose=verbose)) if return_pipeline else dataframe


def auto_preprocess_features_dataframe(dataframe: DataFrame,
                                       return_pipeline: bool = False,
                                       verbose: bool = True) -> Union[DataFrame,
                                                                      (DataFrame, PreprocessPipeline),
                                                                      PreprocessedDataFrame,
                                                                      (PreprocessedDataFrame, PreprocessPipeline)]:
    id_col = 0
    cumulative_binning_value = 'Other'
    divider = '|'
    cumulative_percent = 70

    tools.is_dataframe(dataframe)

    if dataframe.columns.size < 2:
        raise ValueError("Features DataFrame must have id column and at least one feature column")

    if tools.has_na(dataframe, verbose):
        dataframe = DropNa().fit_transform(dataframe)

    if tools.has_duplicates(dataframe, tools.get_df(dataframe).columns[id_col], verbose):
        dataframe = DropDuplicates(subset_features=tools.get_df(dataframe).columns[id_col]).fit_transform(dataframe)

    for feature in tools.get_df(dataframe).iloc[:, (id_col + 1):].columns:
        if tools.is_not_scaled(tools.get_df(dataframe)[feature], verbose):
            dataframe = MinMaxScaler(features=feature).fit_transform(dataframe)
        if tools.is_categorical(tools.get_df(dataframe)[feature], verbose):
            if tools.is_multi_categorical(tools.get_df(dataframe)[feature], divider, verbose):
                if verbose:
                    print(f"Applied BinCumulative at 70% to feature '{feature}' "
                          f"with new value '{cumulative_binning_value}' and using '{divider}' as divider")
                dataframe = BinCumulative(feature,
                                          cumulative_binning_value,
                                          cumulative_threshold=cumulative_percent,
                                          divider=divider).fit_transform(dataframe)
                if verbose:
                    print(f"Applied OneHotEncode transformation to feature \'{feature}\' using '{divider}' as divider")
                dataframe = OneHotEncode(feature, divider=divider).fit_transform(dataframe)
            else:
                if verbose:
                    print(f"Applied BinCumulative at 70% to feature '{feature}' "
                          f"with new value '{cumulative_binning_value}'")
                dataframe = BinCumulative(feature,
                                          cumulative_binning_value,
                                          cumulative_threshold=cumulative_percent).fit_transform(dataframe)

    if isinstance(dataframe, PreprocessedDataFrame):
        if return_pipeline:
            return dataframe, PreprocessPipeline(dataframe.preprocess_functions)
        return dataframe
    return (dataframe, PreprocessPipeline([], verbose=verbose)) if return_pipeline else dataframe
