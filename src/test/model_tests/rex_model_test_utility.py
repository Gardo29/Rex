from itertools import product, groupby
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

from test.base_datasets_test import BaseDatasetsTest


class RexModelTestUtility(BaseDatasetsTest):
    def setUp(self) -> None:
        super(RexModelTestUtility, self).setUp()
        self._random_state = 1234
        self._dataset = self._dataset.drop('timestamp', axis=1)
        self._testset = self._dataset.sample(300, random_state=self._random_state)
        self._dataset_no_weights = self._dataset.drop(columns='rating')
        self._id_pairs = list(
            product(np.sort(np.unique(self._testset.iloc[:, self.USER_ID].values)),
                    np.sort(np.unique(self._testset.iloc[:, self.ITEM_ID].values))))
        self._k = 20
        self._evaluation = make_scorer(lambda x: np.random.randint(0, 100))

    def _manage_predictions(self, predictions, key, sort_function, map_function):
        filtered_prediction = {}
        # for each user -> grouped predictions based on user
        for user_id, group in groupby(sorted(predictions, key=key), key):
            # sort by score
            sorted_group = sorted(list(group), key=sort_function, reverse=True)
            # select k
            k_group = sorted_group[:self._k]
            # remove score
            mapped_group = list(map(map_function, k_group))
            # add to all predictions
            filtered_prediction[user_id] = mapped_group
        # sort result based on user id and then item id
        return pd.DataFrame(filtered_prediction)
