from __future__ import annotations

from itertools import product
from typing import Union, Optional

from numpy.random import RandomState
from pandas import DataFrame
from sklearn import model_selection
from surprise import Dataset, Reader

from rex import preprocessing, tools


def train_anti_test_split(dataset: DataFrame | preprocessing2.PreprocessedDataFrame,
                          default_value: Optional[float | int] = None) -> Union[(DataFrame, DataFrame),
                                                                                (preprocessing2.PreprocessedDataFrame,
                                                                                 preprocessing2.PreprocessedDataFrame)]:
    assert isinstance(default_value, (int, float)) or default_value is None, \
        f"'default_value' must be either float, int or None"
    tools.check_weights_dataframe(dataset)

    work_dataset = dataset if isinstance(dataset, DataFrame) else dataset.dataframe

    if tools.is_no_weights_dataframe(work_dataset):
        work_dataset = tools.add_weight(work_dataset, tools.DEFAULT_WEIGHT)

    surprise_anti_test_set = Dataset.load_from_df(work_dataset, Reader()) \
        .build_full_trainset() \
        .build_anti_testset(default_value)

    surprise_as_list = [[uid, iid, weight] for uid, iid, weight in surprise_anti_test_set]

    dataframe = DataFrame(surprise_as_list, columns=work_dataset.columns)

    return dataset, dataframe if isinstance(dataset, DataFrame) \
        else preprocessing2.PreprocessedDataFrame(dataframe, dataset.preprocess_functions)


def generate_train_test(dataset: DataFrame | preprocessing2.PreprocessedDataFrame,
                        train_size: int | float = 0.75,
                        random_state: int | RandomState | None = None) -> Union[(DataFrame, DataFrame),
                                                                                (preprocessing2.PreprocessedDataFrame,
                                                                                 preprocessing2.PreprocessedDataFrame)]:
    def split(input_dataset: DataFrame) -> (DataFrame, DataFrame):
        # return model_selection.train_test_split(input_dataset, train_size=train_size, random_state=random_state)
        if isinstance(train_size, int):
            return input_dataset, input_dataset.sample(n=train_size, random_state=random_state)
        if isinstance(train_size, float):
            return input_dataset, input_dataset.sample(frac=train_size, random_state=random_state)

    assert isinstance(train_size, (int, float)), "'train_size' must be either int or float"
    tools.check_is_dataframe_or_preprocessed_dataframe(dataset)
    assert isinstance(random_state, (int, RandomState)) or random_state is None, \
        "'random_state' must be either int or Numpy RandomState or None"

    if isinstance(dataset, DataFrame):
        return split(dataset)
    if isinstance(dataset, preprocessing2.PreprocessedDataFrame):
        train, test = split(dataset.dataframe)
        return (preprocessing2.PreprocessedDataFrame(train, dataset.preprocess_functions),
                preprocessing2.PreprocessedDataFrame(test, dataset.preprocess_functions))
