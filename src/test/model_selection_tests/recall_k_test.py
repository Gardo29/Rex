from itertools import product

import numpy as np
from lightfm import lightfm
from lightfm.data import Dataset
from pandas import DataFrame
from lightfm import evaluation
from rex.model import LightFM
from rex.tools import WEIGHT, ITEM_ID, USER_ID, unique, groupby
from rex.model_evaluation import recall_k
from test.model_tests.rex_model_test_utility import RexModelTestUtility


class RecallKTest(RexModelTestUtility):
    def setUp(self) -> None:
        super(RecallKTest, self).setUp()
        self._model = LightFM(random_state=self._random_state).fit(self._dataset,
                                                                   item_features=self._item_features,
                                                                   user_features=self._user_features)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            recall_k(self._model, 'wrong_testset', 'wrong_predictions')

    def test_metric_result_is_a_dict(self):
        self.assertIsInstance(recall_k(self._model, self._testset), dict)

    def test_metric_result_contains_all_user_ids_from_testset(self):
        precisions = recall_k(self._model, self._testset)
        self.assertTrue(all(user in precisions.keys() for user in unique(self._testset.userId)))

    def test_user_recall(self):
        self._metric_and_check(is_user_prediction=True)

    def test_item_recall(self):
        self._metric_and_check(is_user_prediction=False)

    def _metric_and_check(self, is_user_prediction=True):
        # ---------- check models predicts are equals ----------
        rex_predictions = self._model.predict(self._testset,
                                              k=self._k,
                                              item_features=self._item_features,
                                              user_features=self._user_features,
                                              is_user_prediction=is_user_prediction)

        # train and predict recommendations using original LightFM model
        # create lightfm model
        lightfm_model = lightfm.LightFM(random_state=self._random_state)
        # fit Dataset based on features and data
        lightfm_dataset = Dataset()
        item_features = np.sort(np.unique(list(f'genres:{genre}' for genre in self._item_features['genres'])))
        lightfm_dataset.fit(users=np.sort(np.unique(self._dataset.userId)),
                            items=np.sort(np.unique(self._item_features.title)),
                            item_features=item_features,
                            user_features=np.array(['age']))

        # create interactions matrix and weights matrix instance for data
        interactions, weights = lightfm_dataset.build_interactions(self._dataset.values)
        item_features = lightfm_dataset.build_item_features(
            [(item_id, {f'genres:{feature}': 1})
             for item_id, feature in self._item_features.values]
        )
        user_features = lightfm_dataset.build_user_features(
            [(user_id, {'age': feature_value})
             for user_id, feature_value in self._user_features.values]
        )
        # train the model with features
        lightfm_model.fit(interactions,
                          sample_weight=weights,
                          item_features=item_features,
                          user_features=user_features)

        # get mapping
        user_id_mapping, _, item_id_mapping, _ = lightfm_dataset.mapping()
        # creates all ids for prediction
        predictions_user_ids = np.array([user_id_mapping[user] for user, _ in self._id_pairs])
        unchanged_user_ids = [user for user, _ in self._id_pairs]
        predictions_item_ids = np.array([item_id_mapping[item] for _, item in self._id_pairs])
        unchanged_item_ids = [item for _, item in self._id_pairs]

        # prediction using LightFM model
        lightfm_scores = lightfm_model.predict(predictions_user_ids,
                                               predictions_item_ids,
                                               item_features=item_features,
                                               user_features=user_features)

        # zip together unchanged ids and predictions scores
        total_prediction = list(zip(unchanged_user_ids, unchanged_item_ids, lightfm_scores))

        # create a new array containing user * k pairs of top k item id for each user
        # rex_model.predictions = sorted(rex_model.predictions, key=lambda x: (x[0], x[1]))
        final_lightfm = self._manage_predictions(total_prediction,
                                                 key=lambda x: x[USER_ID] if is_user_prediction else x[ITEM_ID],
                                                 sort_function=lambda x: (
                                                     x[WEIGHT], x[ITEM_ID] if is_user_prediction else x[USER_ID]),
                                                 map_function=lambda x: x[ITEM_ID] if is_user_prediction else x[
                                                     USER_ID])
        self.assertEqual(final_lightfm, rex_predictions)

        # ---------- compute precisions ----------
        csr_testset, _ = lightfm_dataset.build_interactions(self._testset.values)
        if is_user_prediction:
            light_fm_precision = evaluation.recall_at_k(lightfm_model,
                                                        csr_testset,
                                                        k=self._k,
                                                        item_features=item_features,
                                                        user_features=user_features).mean()
        else:
            id_pairs = list(product(self._model.user_ids_, np.sort(unique(self._testset.values[:, ITEM_ID]))))
            user_ids = [user_id for user_id, _ in id_pairs]
            item_ids = [item_id for _, item_id in id_pairs]
            mapped_user_ids = [user_id_mapping[user_id] for user_id, _ in id_pairs]
            mapped_item_ids = [item_id_mapping[item_id] for _, item_id in id_pairs]
            predictions = list(zip(user_ids,
                                   item_ids,
                                   lightfm_model.predict(
                                       np.array(mapped_user_ids),
                                       np.array(mapped_item_ids),
                                       user_features=user_features,
                                       item_features=item_features,
                                   )))
            real_values = {movie_id: set(user_ids.iloc[:, USER_ID].tolist())
                           for movie_id, user_ids in self._testset.groupby('movieId')}
            precisions = {}
            for item_id, group in groupby(predictions, lambda x: x[ITEM_ID]).items():
                k_sorted_group = sorted(group, key=lambda x: (x[WEIGHT], x[USER_ID]), reverse=True)[:self._k]
                mapped = list(map(lambda x: x[USER_ID], k_sorted_group))
                precisions[item_id] = len(set(mapped) & real_values[item_id]) / len(real_values[item_id])
            light_fm_precision = np.array(list(precisions.values())).mean()

        rex_precision = np.array(list(recall_k(self._model,
                                               self._testset,
                                               k=self._k,
                                               item_features=self._item_features,
                                               is_user_prediction=is_user_prediction,
                                               user_features=self._user_features).values())).mean()
        # ---------- check precisions are equal ----------

        self.assertAlmostEqual(light_fm_precision, rex_precision)
