import threading
from itertools import product, groupby

import numpy as np
import pandas as pd
from lightfm.data import Dataset
from sklearn.utils.estimator_checks import check_estimator

import rex.model_evaluation
from rex.tools import (unique, WEIGHT, USER_ID, ITEM_ID)
from test.model_tests.rex_model_test_utility import RexModelTestUtility
from rex.model import LightFM
from lightfm import lightfm


class LightFMTest(RexModelTestUtility):
    def setUp(self) -> None:
        super(LightFMTest, self).setUp()
        self._default_params = {'no_components': 10,
                                'k': 5,
                                'n': 10,
                                'learning_schedule': 'adagrad',
                                'loss': 'logistic',
                                'learning_rate': 0.05,
                                'rho': 0.95,
                                'epsilon': 1e-06,
                                'item_alpha': 0.0,
                                'user_alpha': 0.0,
                                'max_sampled': 10,
                                'random_state': None}

    def test_ok_no_arguments(self):
        self.not_fallible_test(lambda: LightFM())

    def test_ok_all_arguments(self):
        self.not_fallible_test(lambda: LightFM(**self._default_params))

    def test_ok_set_params(self):
        self.not_fallible_test(lambda: LightFM().set_params(**self._default_params))

    def test_fit(self):
        self.not_fallible_test(lambda: LightFM().fit(self._dataset, verbose=False))

    def test_fit_item_features(self):
        self.not_fallible_test(lambda: LightFM().fit(self._dataset, verbose=False, item_features=self._item_features))

    def test_fit_wrong_dataset(self):
        with self.assertRaises(ValueError):
            LightFM().fit("wrong dataset")

    def test_predict_is_dataset_length_is_k(self):
        model = LightFM().fit(self._dataset)
        try:
            predicted = model.predict(self._testset, k=self._k)
            if not isinstance(predicted, pd.DataFrame):
                self.fail()
            if len(predicted) != self._k and len(unique(self._testset.values[:, ITEM_ID])) > self._k:
                self.fail()
        except:
            self.fail()

    def test_same_predictions_no_features_no_weights(self):
        self._predict(self._dataset_no_weights, features=False)

    def test_same_predictions_no_features_user(self):
        self._predict(self._dataset, features=False)

    def test_same_predictions_features_user(self):
        self._predict(self._dataset, features=True)

    def test_same_predictions_no_features_item(self):
        self._predict(self._dataset, features=False, is_user_prediction=False)

    def test_same_predictions_features_item(self):
        self._predict(self._dataset, features=True, is_user_prediction=False)

    def test_previous_interactions_user(self):
        self._predict(self._dataset, features=True, previous_interactions=self._dataset)

    def test_previous_interactions_item(self):
        self._predict(self._dataset, features=True, previous_interactions=self._dataset, is_user_prediction=False)

    def test_error_complementary_no_features(self):
        model = self._full_model()
        with self.assertRaises(ValueError):
            model.predict(self._testset,
                          user_features=self._user_features,
                          exclude_features='genre:Comedy')
        with self.assertRaises(ValueError):
            model.predict(self._testset,
                          item_features=self._item_features,
                          exclude_features='age',
                          is_user_prediction=False)

    def test_complementary_feature(self):
        model = self._full_model()
        normal_predictions = model.predict(self._testset,
                                           k=self._k,
                                           item_features=self._item_features,
                                           user_features=self._user_features)
        # select first film
        first_movies = unique(normal_predictions.iloc[0].values)
        genres_to_exclude = self._item_features[self._item_features.title.isin(first_movies)].genres.values

        filtered_predictions = model.predict(self._testset,
                                             k=self._k,
                                             item_features=self._item_features,
                                             user_features=self._user_features,
                                             exclude_features=genres_to_exclude)
        predicted_movie = [movie_id for user_id in filtered_predictions for movie_id in filtered_predictions[user_id]]
        movie_genres = unique(self._item_features[self._item_features.title.isin(predicted_movie)].genres.values)
        self.assertFalse(any(movie_genre in genres_to_exclude for movie_genre in movie_genres))

    def check_sklearn_compatibility(self):
        self.not_fallible_test(lambda: check_estimator(self))

    def _predict(self, dataset, features=True, previous_interactions=None, is_user_prediction=True):
        # train and predict recommendations using Rex model
        rex_model = self._full_model() if features else LightFM(random_state=self._random_state).fit(dataset)
        rex_predictions = rex_model.predict(self._testset,
                                            k=self._k,
                                            is_user_prediction=is_user_prediction,
                                            previous_interactions=previous_interactions,
                                            item_features=self._item_features,
                                            user_features=self._user_features) if features \
            else rex_model.predict(self._testset,
                                   k=self._k,
                                   previous_interactions=previous_interactions,
                                   is_user_prediction=is_user_prediction)

        # train and predict recommendations using original LightFM model
        # create lightfm model
        lightfm_model = lightfm.LightFM(random_state=self._random_state)
        # fit Dataset based on features and data
        lightfm_dataset = Dataset()
        item_features = np.sort(np.unique(list(f'genres:{genre}' for genre in self._item_features['genres'])))
        lightfm_dataset.fit(users=np.sort(np.unique(dataset.userId)),
                            items=np.sort(np.unique(self._item_features.title)) if features else np.sort(
                                np.unique(dataset.movieId)),
                            item_features=item_features if features else None,
                            user_features=np.array(['age']) if features else None)

        # create interactions matrix and weights matrix instance for data
        interactions, weights = lightfm_dataset.build_interactions(dataset.values)
        if features:
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
        else:
            lightfm_model.fit(interactions, sample_weight=weights)

        if previous_interactions is not None:
            to_exclude = [(user_id, item_id) for user_id, item_id in previous_interactions.values[:, :2]]
            self._id_pairs = list(set(self._id_pairs) - set(to_exclude))

        # get mapping
        user_id_mapping, _, item_id_mapping, _ = lightfm_dataset.mapping()
        # creates all ids for prediction
        predictions_user_ids = np.array([user_id_mapping[user] for user, _ in self._id_pairs])
        unchanged_user_ids = [user for user, _ in self._id_pairs]
        predictions_item_ids = np.array([item_id_mapping[item] for _, item in self._id_pairs])
        unchanged_item_ids = [item for _, item in self._id_pairs]

        # prediction using LightFM model
        if features:
            lightfm_scores = lightfm_model.predict(predictions_user_ids,
                                                   predictions_item_ids,
                                                   item_features=item_features,
                                                   user_features=user_features)
        else:
            lightfm_scores = lightfm_model.predict(predictions_user_ids, predictions_item_ids)
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

    def _full_model(self):
        return LightFM(random_state=self._random_state).fit(self._dataset,
                                                            item_features=self._item_features,
                                                            user_features=self._user_features)
