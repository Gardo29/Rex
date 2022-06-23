from itertools import groupby, product

import numpy as np
from lightfm import lightfm
from lightfm.data import Dataset
from pandas import DataFrame
from lightfm import evaluation
from rex.model import LightFM
from rex.tools import WEIGHT, ITEM_ID, USER_ID, unique
from rex.model_evaluation import precision_k
from test.model_tests.rex_model_test_utility import RexModelTestUtility


class PrecisionKTest(RexModelTestUtility):
    def setUp(self) -> None:
        super(PrecisionKTest, self).setUp()
        self._model = LightFM(random_state=self._random_state).fit(self._dataset,
                                                                   item_features=self._item_features,
                                                                   user_features=self._user_features)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            precision_k(self._model, 'wrong_testset', 'wrong_predictions')

    def test_metric_result_is_a_dict(self):
        self.assertIsInstance(precision_k(self._model, self._testset), dict)

    def test_metric_result_contains_all_user_ids_from_testset(self):
        precisions = precision_k(self._model, self._testset)
        self.assertTrue(all(user in precisions.keys() for user in unique(self._testset.userId)))

    def test_user_precision(self):
        self._metric_and_check(is_user_prediction=True)

    def test_item_precision(self):
        self._metric_and_check(is_user_prediction=False)

    def _metric_and_check(self, is_user_prediction=True):
        # ---------- check models predicts are equals ----------
        lightfm_dataframe = self._dataset if is_user_prediction else DataFrame({
            'userId': self._dataset.movieId,
            'movieId': self._dataset.userId,
            'rating': self._dataset.rating
        })
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
        user_features = np.array(['age'])
        user_ids = np.sort(unique(self._user_features.userId))
        movie_ids = np.sort(unique(self._item_features.title))
        lightfm_dataset.fit(users=user_ids if is_user_prediction else movie_ids,
                            items=movie_ids if is_user_prediction else user_ids,
                            item_features=item_features if is_user_prediction else user_features,
                            user_features=user_features if is_user_prediction else item_features)

        # create interactions matrix and weights matrix instance for data
        interactions, weights = lightfm_dataset.build_interactions(lightfm_dataframe.values)

        def item_features():
            return [(item_id, {f'genres:{feature}': 1})
                    for item_id, feature in self._item_features.values]

        def user_features():
            return [(user_id, {'age': feature_value})
                    for user_id, feature_value in self._user_features.values]

        user_features = lightfm_dataset.build_user_features(user_features() if is_user_prediction else item_features())
        item_features = lightfm_dataset.build_item_features(item_features() if is_user_prediction else user_features())
        # train the model with features
        lightfm_model.fit(interactions,
                          sample_weight=weights,
                          item_features=item_features,
                          user_features=user_features)
        # get mapping
        user_id_mapping, _, item_id_mapping, _ = lightfm_dataset.mapping()
        self._id_pairs = list(product(np.sort(unique(lightfm_dataframe.userId)),
                                      np.sort(unique(lightfm_dataframe.movieId))))
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
                                                 key=lambda x: x[USER_ID],
                                                 sort_function=lambda x: (x[WEIGHT], x[ITEM_ID]),
                                                 map_function=lambda x: x[ITEM_ID])
        self.assertEqual(final_lightfm, rex_predictions)

        # ---------- compute precisions ----------
        csr_testset, _ = lightfm_dataset.build_interactions(self._testset.values)
        light_fm_precision = evaluation.precision_at_k(lightfm_model,
                                                       csr_testset,
                                                       k=self._k,
                                                       item_features=item_features,
                                                       user_features=user_features).mean()
        rex_precision = np.array(list(precision_k(self._model,
                                                  self._testset,
                                                  k=self._k,
                                                  item_features=self._item_features,
                                                  user_features=self._user_features,
                                                  ).values())).mean()
        # ---------- check precisions are equal ----------

        self.assertAlmostEqual(light_fm_precision, rex_precision)
