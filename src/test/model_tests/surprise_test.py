from abc import abstractmethod, ABC
from typing import Dict, Type

import numpy as np
import surprise
from sklearn.model_selection import GridSearchCV
from sklearn.utils.estimator_checks import check_estimator
from surprise import Reader, AlgoBase

from rex.model import SurpriseModel
from test.model_tests.rex_model_test_utility import RexModelTestUtility
from surprise.dataset import Dataset


# TODO grid search
class SurpriseTest(RexModelTestUtility, ABC):

    def setUp(self) -> None:
        super(SurpriseTest, self).setUp()
        reader = Reader(rating_scale=(self._dataset.rating.min(), self._dataset.rating.min()))
        self._surprise_dataset = Dataset.load_from_df(self._dataset, reader).build_full_trainset()

    @abstractmethod
    def _rex_algo(self) -> type:
        pass

    @abstractmethod
    def _surprise_algo(self) -> type:
        pass

    @abstractmethod
    def _get_default_params(self) -> dict:
        pass

    @abstractmethod
    def _grid_search_parameters(self) -> dict:
        pass

    def test_ok_no_arguments(self):
        self.not_fallible_test(lambda: self._rex_algo()())

    def test_ok_all_arguments(self):
        self.not_fallible_test(lambda: self._rex_algo()(**self._get_default_params()))

    def test_ok_set_params(self):
        self.not_fallible_test(lambda: self._rex_algo()().set_params(**self._get_default_params()))

    def test_fit(self):
        self._rex_algo()().fit(self._dataset, verbose=False)

    def test_fit_no_weights(self):
        self._rex_algo()().fit(self._dataset_no_weights, verbose=False)

    def test_fit_wrong_dataset(self):
        with self.assertRaises(ValueError):
            self._rex_algo()().fit("wrong dataset")

    def test_check_sklearn_compatibility(self):
        self.not_fallible_test(lambda: check_estimator(self._rex_algo()()))

    def test_check_exclude_train_from_predictions_user(self):
        model = self._rex_algo()().fit(self._dataset)
        user_predictions = model.predict(self._testset, previous_interactions=self._dataset)
        predictions_pairs = RexModelTestUtility.to_pairs(user_predictions)
        dataset_pairs = {(user_id, item_id) for user_id, item_id in self._dataset.values[:, :2]}
        self.assertEqual(predictions_pairs & dataset_pairs, set())

    def test_check_exclude_train_from_predictions_item(self):
        model = self._rex_algo()().fit(self._dataset)
        item_predictions = model.predict(self._testset, previous_interactions=self._dataset, is_user_prediction=False)
        predictions_pairs = RexModelTestUtility.to_pairs(item_predictions, is_user_prediction=False)
        dataset_pairs = {(user_id, item_id) for user_id, item_id in self._dataset.values[:, :2]}
        print(predictions_pairs & dataset_pairs)
        self.assertEqual(predictions_pairs & dataset_pairs, set())

    def test_grid_search_cv(self):
        grid = GridSearchCV(self._rex_algo()(), self._grid_search_parameters(), scoring=self._evaluation)
        # grid.fit(self._dataset)
        # grid.predict(self._testset, k=self._k)

    def _check_equals_predict_user(self, random_state=True):
        rex_model = self._rex_algo()(random_state=self._random_state) if random_state else self._rex_algo()()
        rex_predictions = rex_model.fit(self._dataset, verbose=False).predict(self._testset, k=self._k)

        surprise_model = self._surprise_algo()(random_state=self._random_state) if random_state else \
            self._surprise_algo()()
        surprise_model.fit(self._surprise_dataset)
        surprise_predictions = [surprise_model.predict(uid, iid, clip=False, verbose=False)
                                for uid, iid in self._id_pairs]
        final_surprise = self._manage_predictions(surprise_predictions,
                                                  lambda x: x.uid,
                                                  lambda x: (x.est, x.iid),
                                                  lambda x: x.iid)

        self.assertEqual(final_surprise, rex_predictions)

    def _check_equals_predict_item(self, random_state=True):
        rex_model = self._rex_algo()(random_state=self._random_state) if random_state else self._rex_algo()()
        rex_predictions = rex_model.fit(self._dataset, verbose=False).predict(self._testset, k=self._k,
                                                                              is_user_prediction=False)

        surprise_model = self._surprise_algo()(random_state=self._random_state) if random_state else \
            self._surprise_algo()()
        surprise_model.fit(self._surprise_dataset)
        surprise_predictions = [surprise_model.predict(uid, iid, clip=False, verbose=False)
                                for uid, iid in self._id_pairs]
        final_surprise = self._manage_predictions(surprise_predictions,
                                                  lambda x: x.iid,
                                                  lambda x: (x.est, x.uid),
                                                  lambda x: x.uid)

        self.assertEqual(final_surprise, rex_predictions)
