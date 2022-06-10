from __future__ import annotations

from typing import Union

from numpy.random import RandomState
from pandas import DataFrame
from sklearn import model_selection

from rex import preprocessing2


def train_test_split(dataset: DataFrame | preprocessing2.PreprocessedDataFrame,
                     train_size: int | float = 0.75,
                     random_state: int | RandomState | None = None) -> Union[(DataFrame, DataFrame),
                                                                             (preprocessing2.PreprocessedDataFrame,
                                                                              preprocessing2.PreprocessedDataFrame)]:
    def split(input_dataset: DataFrame) -> (DataFrame, DataFrame):
        return model_selection.train_test_split(input_dataset, train_size=train_size, random_state=random_state)

    if isinstance(dataset, DataFrame):
        return split(dataset)
    if isinstance(dataset, preprocessing2.PreprocessedDataFrame):
        train, test = split(dataset.dataframe)
        return (preprocessing2.PreprocessedDataFrame(train, dataset.preprocess_functions),
                preprocessing2.PreprocessedDataFrame(test, dataset.preprocess_functions))
    else:
        raise ValueError("'dataset' must be either Pandas DataFrame or PreprocessedDataFrame")
