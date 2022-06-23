from itertools import product
from typing import Callable, Optional
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

from rex.model_evaluation import precision_k
from rex.tools import USER_ID, ITEM_ID, groupby
from test.base_datasets_test import BaseDatasetsTest


class RexModelTestUtility(BaseDatasetsTest):
    def setUp(self) -> None:
        super(RexModelTestUtility, self).setUp()
        self._dataset = self._dataset.drop('timestamp', axis=1)
        self._testset = self._dataset.sample(300, random_state=self._random_state)
        self._dataset_no_weights = self._dataset.drop(columns='rating')
        self._id_pairs = list(product(np.sort(np.unique(self._testset.iloc[:, USER_ID].values)),
                                      np.sort(np.unique(self._testset.iloc[:, ITEM_ID].values))))
        self._k = 400
        self._evaluation = make_scorer(precision_k)

    def _manage_predictions(self,
                            predictions,
                            key,
                            sort_function,
                            map_function) -> pd.DataFrame:
        def select_predictions(grouped_predictions):
            # sort predictions by ranking
            sorted_predictions = sorted(grouped_predictions, key=sort_function, reverse=True)
            # select first k
            k_predictions = sorted_predictions[:k]
            # extract only ids
            mapped_predictions = map(map_function, k_predictions)
            return list(mapped_predictions)

        # compute groups
        groups = {key_id: group for key_id, group in groupby(predictions, key).items()}
        k = min(min(len(group_list) for group_list in groups.values()), self._k)

        # group by key, and extract for each id top k predictions
        managed_predictions = {key_id: select_predictions(grouped_predictions)
                               for key_id, grouped_predictions in groups.items()}

        # pd.testing.assert_frame_equal(pd.DataFrame(managed_predictions), pd.DataFrame(rex_predictions))
        return pd.DataFrame(managed_predictions).sort_index(axis=1)

    @staticmethod
    def to_pairs(dataframe, is_user_prediction=True):
        return {(id1, id2) if is_user_prediction else (id2, id1) for id1 in dataframe for id2 in dataframe[id1]}
