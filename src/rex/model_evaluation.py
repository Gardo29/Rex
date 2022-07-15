from __future__ import annotations

from typing import Optional

from pandas import DataFrame
import numpy as np
from rex import model
from rex.preprocessing import PreprocessedDataFrame
from rex.tools import (check_is_dataframe_or_preprocessed_dataframe,
                       unique,
                       ITEM_ID,
                       USER_ID)


def _predict(rex_model: model.RexBaseModel,
             test_interactions: DataFrame | PreprocessedDataFrame,
             train_interactions: Optional[DataFrame | PreprocessedDataFrame] = None,
             k=10,
             is_user_prediction: bool = True,
             **kwargs) -> tuple:
    to_predict = USER_ID if is_user_prediction else ITEM_ID  # interest ids
    to_be_predicted = USER_ID if not is_user_prediction else ITEM_ID  # interest ids

    assert k > 0, f"'k' must be positive"
    check_is_dataframe_or_preprocessed_dataframe(test_interactions)
    # extract dataframe in case
    if isinstance(test_interactions, PreprocessedDataFrame):
        test_interactions = test_interactions.dataframe

    if train_interactions:
        check_is_dataframe_or_preprocessed_dataframe(train_interactions)

    # calculate ids to be used in predict
    user_ids = np.sort(unique(test_interactions.iloc[:, USER_ID])) if is_user_prediction else rex_model.user_ids_
    item_ids = np.sort(unique(test_interactions.iloc[:, ITEM_ID])) if not is_user_prediction else rex_model.item_ids_
    predictions = rex_model.predict(user_ids, item_ids, k, train_interactions, is_user_prediction, **kwargs)

    condensed_predictions = {to_be_predicted_id: set(predictions[to_be_predicted_id].tolist())
                             for to_be_predicted_id in predictions}

    condensed_real_values = {main_id: set(condensed_ids.iloc[:, to_be_predicted].tolist())
                             for main_id, condensed_ids
                             in test_interactions.groupby(test_interactions.columns[to_predict])}

    return condensed_real_values, condensed_predictions


def recall_k(rex_model: model.RexBaseModel,
             test_interactions: DataFrame | PreprocessedDataFrame,
             train_interactions: Optional[DataFrame | PreprocessedDataFrame] = None,
             k=10,
             is_user_prediction: bool = True,
             **kwargs) -> dict:
    real, predicted = _predict(rex_model, test_interactions, train_interactions, k, is_user_prediction, **kwargs)
    recalls = {}
    for main_id, real_values in real.items():
        hit = len(real_values & predicted[main_id])
        retrieved = len(real_values)
        recalls[main_id] = hit / retrieved
    return recalls


def precision_k(rex_model: model.RexBaseModel,
                test_interactions: DataFrame | PreprocessedDataFrame,
                train_interactions: Optional[DataFrame | PreprocessedDataFrame] = None,
                k=10,
                is_user_prediction: bool = True,
                **kwargs) -> dict:
    real, predicted = _predict(rex_model, test_interactions, train_interactions, k, is_user_prediction, **kwargs)

    return {main_id: len((real_values & predicted[main_id])) / k
            for main_id, real_values in real.items()}
