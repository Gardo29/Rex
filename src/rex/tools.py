from __future__ import annotations
from typing import Iterable, Union
from matplotlib import gridspec
from pandas.api.types import (is_integer_dtype, is_float_dtype)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from sklearn import preprocessing

from rex import preprocessing2


# --------------- UTILITY --------------------
def unique(array: np.ndarray | list):
    if isinstance(array, list):
        array = np.array(array)
    if is_integer_dtype(array.dtype) or is_float_dtype(array.dtype):
        return np.unique(array)
    else:
        return list(set(array))


# ------------- DATAFRAME LOADING -------------

def to_dataframe(data, column_names=None):
    return DataFrame(data, columns=column_names)


def to_dataframe_utility_matrix(weights, user_ids=None, items_ids=None):
    return DataFrame(weights, index=user_ids, columns=items_ids)


# ---- DATAFRAME EXTRACTOR -------------
def get_df(dataframe: DataFrame | preprocessing2.PreprocessedDataFrame) -> DataFrame:
    if isinstance(dataframe, DataFrame):
        return dataframe
    elif isinstance(dataframe, preprocessing2.PreprocessedDataFrame):
        return dataframe.dataframe
    else:
        raise ValueError


# ---------------- CHECKS ---------------
def is_dataframe(dataframe: preprocessing2.PreprocessedDataFrame | DataFrame) -> None:
    if not isinstance(dataframe, DataFrame) and not isinstance(dataframe, preprocessing2.PreprocessedDataFrame):
        raise ValueError('Input data must be either a Pandas DataFrame or PreprocessedDataFrame')


def is_no_weights_dataframe(dataframe: preprocessing2.PreprocessedDataFrame | DataFrame):
    is_dataframe(dataframe)
    if isinstance(dataframe, preprocessing2.PreprocessedDataFrame):
        return dataframe.dataframe.columns.size == 2
    else:
        return dataframe.columns.size == 2


def check_weights_dataframe(dataset: DataFrame | preprocessing2.PreprocessedDataFrame) -> None:
    user_id_column = 0
    item_id_column = 1
    weights_column = 2
    min_columns = 2
    max_columns = 3
    is_dataframe(dataset)
    if not (min_columns <= get_df(dataset).columns.size <= max_columns):
        raise ValueError("Pandas DataFrame must have 2 or 3 columns")
    if get_df(dataset).columns.size == max_columns and not is_numerical(get_df(dataset).iloc[:, weights_column],
                                                                        verbose=False):
        raise ValueError("Weights column, if present, must contain either int or float values")
    if has_na(dataset):
        raise ValueError("DataFrame can't contain NaN values")
    if has_duplicates(dataset, get_df(dataset).columns[[user_id_column, item_id_column]].tolist()):
        raise ValueError("DataFrame can't contain duplicates")


def check_features(dataframe: DataFrame | preprocessing2.PreprocessedDataFrame):
    min_columns = 2
    feature_id_column = 0
    is_dataframe(dataframe)
    if get_df(dataframe).columns.size < min_columns:
        raise ValueError("Pandas DataFrame must have at least 2 columns")
    if has_na(dataframe):
        raise ValueError("DataFrame can't contain NaN values")
    if has_duplicates(dataframe, get_df(dataframe).columns[feature_id_column]):
        raise ValueError("DataFrame can't contain duplicates")


def is_multi_categorical(series, divider, verbose=True):
    if is_categorical(series, False) and series.str.contains(divider, regex=False).sum() > 0:
        if verbose:
            print(f"'{series.name}' is multi categorical")
        return True
    return False


def has_na(dataframe: DataFrame | preprocessing2.PreprocessedDataFrame, verbose=True) -> bool:
    nan_is_present = False
    for column in get_df(dataframe):
        if get_df(dataframe)[column].isna().sum() > 0:
            if verbose:
                print(f"DataFrame has Nan values in column '{column}'")
            nan_is_present = True
    return nan_is_present


def is_categorical(series, verbose=True) -> bool:
    if not is_float_dtype(series.dtype) and not is_integer_dtype(series.dtype):
        if verbose:
            print(f"'{series.name}' is categorical")
        return True
    return False


def is_numerical(series, verbose=True) -> bool:
    if is_float_dtype(series.dtype) or is_integer_dtype(series.dtype):
        if verbose:
            print(f"'{series.name}' has numeric dtype: {'int' if is_float_dtype(series.dtype) else 'float'}")
        return True
    return False


def is_not_scaled(series, verbose=True) -> bool:
    if is_numerical(series, verbose=verbose) and (series.max() > 1 or series.min() < 0):
        if verbose:
            print(f"'{series.name}' is not in [0,1]")
        return True
    return False


def is_not_normalized(series, verbose=True) -> bool:
    if is_numerical(series, verbose=verbose):
        normalized_value = preprocessing.Normalizer().fit_transform(np.atleast_2d(series.values)).flatten()
        return True if not np.array_equal(normalized_value, series.values) else False
    return False


def has_duplicates(dataframe, subset, verbose=True) -> bool:
    if get_df(dataframe).duplicated(subset=subset).sum() > 0:
        if verbose:
            print(f"DataFrame has duplicates in '{subset}'")
        return True
    return False


# ------------- ADVICE -------------
def dataframe_advisor(dataframe: DataFrame,
                      id_columns: str | [str],
                      is_feature_matrix: bool = False,
                      verbose: bool = True) -> None:
    if is_feature_matrix:
        if dataframe.columns.size < 2 and verbose:
            print('DataFrame has less than the minimum of 2 columns')
    else:
        if not (2 <= dataframe.columns.size <= 3) and verbose:
            print(f"DataFrame doesn't a minimum of 2 columns and a maximum of 3. "
                  f"It has {dataframe.columns.size} columns")

    has_na(dataframe, verbose)
    has_duplicates(dataframe, id_columns, verbose)

    for column in dataframe.drop(id_columns, axis=1):
        is_not_scaled(dataframe[column], verbose)
        is_not_normalized(dataframe[column], verbose)
        is_multi_categorical(dataframe[column], verbose)


# --------------DESCRIBE--------------

def plot_hist(series, bins='auto', ax=None):
    sns.histplot(series, bins=bins, ax=ax)


def plot_continuous_distribution(series, ax=None):
    sns.kdeplot(series, ax=ax)


def plot_categorical_distribution(series, split=None, ax=None):
    series = series.str.split(split).explode() if split is not None else series
    grouped_values = series.value_counts()
    grouped_values_percent = series.value_counts(normalize=True) * 100
    groups_info = zip(grouped_values.index, grouped_values.values, grouped_values_percent.values)
    label_string = "{}  ({} - {:.2f}%)"

    plt.title(f"""Number of features: {grouped_values.index.size}
                      Total values: {grouped_values.sum()}""")
    labels = [label_string.format(index, value, percentage) for index, value, percentage in groups_info]
    sns.barplot(y=labels, x=grouped_values.values, orient='h', ax=ax)


def describe(dataframe, display_dataframe=True, categorical=None, continuous=None, hist=None):
    current_rows = 0
    total_figure = plt.figure(figsize=(10, 8))

    def add_plot(rows, fig):
        new_rows = rows + 1
        gs = gridspec.GridSpec(new_rows, 1)
        for i, ax in enumerate(fig.axes):
            ax.set_position(gs[i].get_position(fig))
            ax.set_subplotspec(gs[i])
        return new_rows, fig.add_subplot(gs[new_rows - 1])

    if display_dataframe:
        print(dataframe)

    if categorical is not None:
        if isinstance(categorical, dict):
            for feature, split in categorical.items():
                current_rows, new_ax = add_plot(current_rows, total_figure)
                plot_categorical_distribution(dataframe[feature], split=split, ax=new_ax)
        elif isinstance(categorical, str) or isinstance(categorical, Iterable):
            categorical = np.atleast_1d(categorical)
            for feature in categorical:
                current_rows, new_ax = add_plot(current_rows, total_figure)
                plot_categorical_distribution(dataframe[feature], ax=new_ax)

    if continuous is not None:
        continuous = np.atleast_1d(continuous)
        for feature in continuous:
            current_rows, new_ax = add_plot(current_rows, total_figure)
            plot_continuous_distribution(dataframe[feature], ax=new_ax)

    if hist is not None:
        if isinstance(hist, dict):
            for feature, bins in hist.items():
                current_rows, new_ax = add_plot(current_rows, total_figure)
                plot_hist(dataframe[feature], bins, ax=new_ax)
        elif isinstance(hist, str) or isinstance(hist, Iterable):
            hist = np.atleast_1d(hist)
            for feature in hist:
                current_rows, new_ax = add_plot(current_rows, total_figure)
                plot_hist(dataframe[feature], ax=new_ax)

    plt.show()
