from __future__ import annotations

import re
from typing import (Union, Optional, Any, Callable, Iterable, TypeVar)
import itertools
from abc import abstractmethod, ABC
from collections import defaultdict
from itertools import (product)  # for surprise predict
from time import time
import threading
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import rex.tools
from rex import model_evaluation
from rex.preprocessing2 import PreprocessedDataFrame, auto_preprocess_weights_dataframe, PreprocessPipeline, \
    auto_preprocess_features_dataframe
from rex.tools import (check_weights_dataframe,
                       check_is_dataframe_or_preprocessed_dataframe,
                       check_features,
                       unique,
                       groupby,
                       dataframe_advisor,
                       is_no_weights_dataframe,
                       WEIGHT,
                       ITEM_ID,
                       USER_ID,
                       CATEGORICAL_FEATURE_WEIGHT,
                       FEATURE_ID, DEFAULT_WEIGHT, add_weight)
# LightFM
from lightfm import data
import lightfm
# surprise
import surprise
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise import Reader, Prediction
from surprise import prediction_algorithms


# ---------------- UTILITY FUNCTIONS --------------
def dataframe_to_id_pairs(dataframe: DataFrame) -> list[(Any, Any)]:
    return [(user_id, item_id) for user_id, item_id in dataframe.values[:, :WEIGHT]]


def compute_predictions(predictions,
                        k: int,
                        key: Callable,
                        sort_function: Callable,
                        map_function: Callable,
                        filter_function: Optional[Callable] = None) -> DataFrame:
    def select_predictions(grouped_predictions):
        # in case filter by function
        if filter_function:
            grouped_predictions = filter(filter_function, grouped_predictions)
        # sort predictions by ranking
        sorted_predictions = sorted(grouped_predictions, key=sort_function, reverse=True)
        # select first k
        k_predictions = sorted_predictions[:k]
        # extract only ids
        mapped_predictions = map(map_function, k_predictions)
        return list(mapped_predictions)

    # compute groups
    groups = {key_id: group for key_id, group in groupby(predictions, key).items()}
    k = min(min(len(group_list) for group_list in groups.values()), k)

    # group by key, and extract for each id top k predictions
    return DataFrame({key_id: select_predictions(grouped_predictions)
                      for key_id, grouped_predictions in groups.items()}).sort_index(axis=1)


# ---------------- MODELS --------------

class RexBaseModel(ABC, BaseEstimator):
    def fit(self, dataset: DataFrame | PreprocessedDataFrame, y=None, verbose: bool = True, **kwargs) -> Any:
        check_weights_dataframe(dataset)
        # checks datasets
        if 'user_features' in kwargs:
            check_features(kwargs['user_features'])
        if 'item_features' in kwargs:
            check_features(kwargs['item_features'])
        # save start time
        start_time = time()
        if verbose:
            print(f"Fitting '{self.__class__.__name__}'")
        # extract dataframe if there are PreprocessedDataFrame
        if isinstance(dataset, PreprocessedDataFrame):
            dataset = dataset.dataframe
        if 'user_features' in kwargs and isinstance(kwargs['user_features'], PreprocessedDataFrame):
            kwargs['user_features'] = kwargs['user_features'].dataframe
        if 'item_features' in kwargs and isinstance(kwargs['item_features'], PreprocessedDataFrame):
            kwargs['item_features'] = kwargs['item_features'].dataframe

        # save uid and iid for checks
        self.user_ids_ = np.sort(unique(dataset.iloc[:, USER_ID]))
        self.item_ids_ = np.sort(unique(dataset.iloc[:, ITEM_ID]))
        # fit the model
        self._checked_fit(dataset, verbose, **kwargs)
        if verbose:
            print(f"Fit time '{self.__class__.__name__}': {time() - start_time}s")
        return self

    @abstractmethod
    def _checked_fit(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> Any:
        pass

    def predict(self,
                x: Iterable | DataFrame | PreprocessedDataFrame | set[tuple],
                item_ids: Optional[Iterable] = None,
                k: int = 10,
                previous_interactions: Optional[DataFrame | PreprocessedDataFrame] = None,
                is_user_prediction: bool = True,
                **kwargs) -> DataFrame:
        # check if model is fitted
        check_is_fitted(self)
        # check k
        assert isinstance(k, int) and k > 0, f"'k' must be a positive integer"
        # check previous interactions
        if previous_interactions is not None:
            check_is_dataframe_or_preprocessed_dataframe(previous_interactions)
            if isinstance(previous_interactions, PreprocessedDataFrame):
                previous_interactions = previous_interactions.dataframe

        # check features if present and extract
        if 'user_features' in kwargs:
            user_features = kwargs['user_features']
            check_features(user_features)
            if isinstance(user_features, PreprocessedDataFrame):
                kwargs['user_features'] = user_features.dataframe
        if 'item_features' in kwargs:
            item_features = kwargs['item_features']
            check_features(kwargs['item_features'])
            if isinstance(item_features, PreprocessedDataFrame):
                kwargs['item_features'] = item_features.dataframe
        # check exclude features
        if 'exclude_features' in kwargs:
            if (is_user_prediction and 'item_features' not in kwargs) or (
                    not is_user_prediction and 'user_features' not in kwargs):
                raise ValueError("If 'exclude_features' is specified relative features must be provided")
            kwargs['exclude_features'] = list(np.atleast_1d(kwargs['exclude_features']))

        # if it's PreprocessedDataFrame extract
        if isinstance(x, PreprocessedDataFrame):
            x = x.dataframe
        # if it's a normal DataFrame check DataFrame and extract ids
        if isinstance(x, DataFrame):
            check_weights_dataframe(x)
            user_ids = np.sort(unique(x.iloc[:, USER_ID].values))
            item_ids = np.sort(unique(x.iloc[:, ITEM_ID].values))
        # if there are ids pairs all work is done, return
        elif isinstance(x, Iterable) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in x):
            user_ids = [uid for uid, _ in x]
            item_ids = [iid for _, iid in x]
        # if there are two ids iterable just pass them
        elif isinstance(x, Iterable) and isinstance(item_ids, Iterable):
            user_ids = np.sort(unique(x))
            item_ids = np.sort(unique(item_ids))
        else:
            raise ValueError(
                "'x' must be either a Pandas Dataframe or "
                "provide both 'x' and 'item_id' as Iterables or "
                "provide a unique Iterable with ids couples")
        # check ids where in train set
        for predict_user_id in user_ids:
            if predict_user_id not in self.user_ids_:
                raise ValueError(f"'{predict_user_id}' not in fit users' ids")
        for predict_item_id in item_ids:
            if predict_item_id not in self.item_ids_:
                raise ValueError(f"'{predict_item_id}' not in fit items' ids")
        # run the predictions
        previous_ids = dataframe_to_id_pairs(previous_interactions) if previous_interactions is not None else []
        id_pairs = list(set(product(user_ids, item_ids)) - set(previous_ids))
        return self._checked_predict(id_pairs, k=k, is_user_prediction=is_user_prediction, **kwargs)

    @abstractmethod
    def _checked_predict(self,
                         id_pairs: list[(Any, Any)],
                         k: int = 10,
                         is_user_prediction: bool = True,
                         **kwargs) -> DataFrame:
        pass

    def set_params(self, **kwargs) -> Any:
        # reset attributes post fit
        fit_attributes_regex = re.compile('.*_$')
        for attribute in filter(fit_attributes_regex.match, self.__dict__):
            if hasattr(self, attribute):
                delattr(self, attribute)
        return self

    @abstractmethod
    def get_params(self, deep=True) -> dict:
        pass


class RexWrapperModel(RexBaseModel, ABC):
    def __init__(self, model_class: Any, **kwargs):
        self._model_class = model_class
        self._model = model_class(**kwargs)

    def set_params(self, **params) -> Any:
        super(RexWrapperModel, self).set_params(**params)
        self._model = self._model_class(**params)
        return self

    def get_params(self, deep=True):
        return self._model.__dict__

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.get_params() == other.get_params()


class SurpriseModel(RexWrapperModel):
    def __init__(self, model, **kwargs):
        self._params = kwargs
        super(SurpriseModel, self).__init__(model, **kwargs)

    def _checked_fit(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> Any:
        if is_no_weights_dataframe(dataset):
            # create Reader for read data from DataFrame no weights
            dataset = add_weight(dataset, DEFAULT_WEIGHT)
            reader = Reader(rating_scale=(1, 1))
        else:
            # select ratings
            ratings = dataset.iloc[:, WEIGHT]
            # create Reader for read data from DataFrame
            reader = Reader(rating_scale=(ratings.min(), ratings.max()))
        # create Surprise dataset from input DataFrame
        surprise_dataset = surprise.Dataset.load_from_df(dataset, reader).build_full_trainset()
        # fit model
        return self._fit_surprise_model(surprise_dataset, verbose, **kwargs)

    @abstractmethod
    def _fit_surprise_model(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> Any:
        pass

    def _checked_predict(self,
                         id_pairs: list[(Any, Any)],
                         k: int = 10,
                         is_user_prediction: bool = True,
                         **kwargs) -> DataFrame:
        # calculates all pairs predictions
        all_predictions = [self._model.predict(user_id, item_id, clip=False, **kwargs)
                           for user_id, item_id in id_pairs]
        # extract top k predictions
        k_top_predictions_dataframe = compute_predictions(
            predictions=all_predictions,
            filter_function=lambda x: not x.details['was_impossible'],
            k=k,
            key=lambda x: x.uid if is_user_prediction else x.iid,
            sort_function=lambda x: (x.est, x.iid if is_user_prediction else x.uid),
            map_function=lambda x: x.iid if is_user_prediction else x.uid
        )
        # create dataframe from predictions dict
        return k_top_predictions_dataframe


class SVD(SurpriseModel):
    def __init__(self, **kwargs):
        super(SVD, self).__init__(surprise.SVD, **kwargs)

    def _fit_surprise_model(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> SVD:
        if verbose:
            self._model = self._model_class(verbose=True, **self._params)
        self._model.fit(dataset, **kwargs)
        # class attributes after training
        self.pu_ = self._model.pu
        self.qi_ = self._model.qi
        self.bu_ = self._model.bu
        self.bi_ = self._model.bi
        return self


class KNNBaseline(SurpriseModel):
    def __init__(self, **kwargs):
        super(KNNBaseline, self).__init__(surprise.KNNBaseline, **kwargs)

    def _fit_surprise_model(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> KNNBaseline:
        self._model = self._model_class(**self._params)
        self._model.fit(dataset, **kwargs)
        # class attributes after training
        self.baseline_ = self._model.compute_baselines()
        return self


class SlopeOne(SurpriseModel):
    def __init__(self):
        super(SlopeOne, self).__init__(surprise.SlopeOne)

    def _fit_surprise_model(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> SlopeOne:
        self._model.fit(dataset)
        # class attributes after training
        self.user_means_ = self._model.user_mean
        return self


class LightFM(RexWrapperModel):
    def __init__(self, **kwargs):
        super(LightFM, self).__init__(lightfm.LightFM, **kwargs)

    def _checked_fit(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> LightFM:
        # extract features data: ids, all possible features, [id,{feature:weight,...}..]
        features_user_ids, all_user_features, user_features = self._extract_features_data(
            kwargs.get('user_features', None))
        features_item_ids, all_item_features, item_features = self._extract_features_data(
            kwargs.get('item_features', None))
        # create Dataset builder
        self._lfm_builder = data.Dataset(
            user_identity_features=kwargs.get('user_identity_features', True),
            item_identity_features=kwargs.get('item_identity_features', True)
        )
        # fit Dataset builder
        self._lfm_builder.fit(
            users=self._concat_unique_list(dataset.iloc[:, USER_ID].values, features_user_ids),
            items=self._concat_unique_list(dataset.iloc[:, ITEM_ID].values, features_item_ids),
            user_features=all_user_features,
            item_features=all_item_features,
        )
        # compute interaction and weights matrices
        interaction_matrix, weights = self._lfm_builder.build_interactions(dataset.values)
        # eventually compute user and item feature matrices
        kwargs['user_features'] = self._lfm_builder.build_user_features(user_features) if user_features else None
        kwargs['item_features'] = self._lfm_builder.build_item_features(item_features) if item_features else None
        # fit LightFM model
        self.interaction_matrix = interaction_matrix
        self.weights = weights
        self.train_user_features = kwargs['user_features']
        self.train_item_features = kwargs['item_features']
        self._model.fit(interaction_matrix,
                        sample_weight=weights,
                        verbose=verbose,
                        **kwargs)
        # save internal mapping for ids
        self._user_id_map, _, self._item_id_map, _ = self._lfm_builder.mapping()
        # class attributes after training
        self.item_embeddings_ = self._model.item_embeddings
        self.user_embeddings_ = self._model.user_embeddings
        self.item_biases_ = self._model.item_biases
        self.user_biases_ = self._model.user_biases
        return self

    def _checked_predict(self,
                         id_pairs: list[(Any, Any)],
                         k: int = 10,
                         is_user_prediction: bool = True,
                         **kwargs) -> DataFrame:

        def compute_exclude_features(features, to_excluded):
            return [feature_id
                    for feature_id, features in features
                    for feature in features.keys()
                    if any(exclude in feature for exclude in to_excluded)]

        # if there are features to be excluded
        if 'exclude_features' in kwargs:
            exclude_features = kwargs['exclude_features']
            # remove them from kwargs
            del kwargs['exclude_features']
            # check consistency
        else:
            exclude_features = None
        # rebuild features matrices and eventually filter for complementary
        if 'user_features' in kwargs:
            _, _, user_features = self._extract_features_data(kwargs.get('user_features'))
            kwargs['user_features'] = self._lfm_builder.build_user_features(user_features)
            # if there are features to be excluded and mode isn't 'user'
            if exclude_features and not is_user_prediction:
                # compute users to be excluded
                user_to_be_excluded = compute_exclude_features(user_features, exclude_features)
                # remove those users from ids_pairs
                id_pairs = {(user_id, item_id) for user_id, item_id in id_pairs if user_id not in user_to_be_excluded}

        if 'item_features' in kwargs:
            _, _, item_features = self._extract_features_data(kwargs.get('item_features'))
            kwargs['item_features'] = self._lfm_builder.build_item_features(item_features)
            # if there are features to be excluded and mode is 'user'
            if exclude_features and is_user_prediction:
                # compute users to be excluded
                item_to_be_excluded = compute_exclude_features(item_features, exclude_features)
                # remove those users from ids_pairs
                id_pairs = {(user_id, item_id) for user_id, item_id in id_pairs if item_id not in item_to_be_excluded}
        # create LightFM ids
        # id_pairs = sorted(list(id_pairs), key=lambda x: (x[0], x[1]))
        unchanged_user_ids = [user_id for user_id, _ in id_pairs]
        lightfm_user_ids = np.array([self._user_id_map[user_id] for user_id in unchanged_user_ids])
        unchanged_item_ids = [item_id for _, item_id in id_pairs]
        lightfm_item_ids = np.array([self._item_id_map[item_id] for item_id in unchanged_item_ids])
        # compute scores
        scores = self._model.predict(lightfm_user_ids, lightfm_item_ids, **kwargs)
        # compute top k predictions
        k_predictions_dataframe = compute_predictions(
            predictions=list(zip(unchanged_user_ids, unchanged_item_ids, scores)),
            k=k,
            key=lambda x: x[USER_ID] if is_user_prediction else x[ITEM_ID],
            sort_function=lambda x: (x[WEIGHT], x[ITEM_ID] if is_user_prediction else x[USER_ID]),
            map_function=lambda x: x[ITEM_ID] if is_user_prediction else x[USER_ID]
        )
        return k_predictions_dataframe

    @staticmethod
    def _concat_unique_list(list1: np.ndarray, list2: Optional[np.ndarray]) -> np.ndarray:
        if list2 is None:
            return np.sort(unique(list1))
        else:
            return np.sort(unique(np.append(list1, list2)))

    @staticmethod
    def _extract_features_data(optional_features: Optional[DataFrame]) -> (Optional[np.ndarray],
                                                                           Optional[set],
                                                                           Optional[list[tuple[Any, dict]]]):
        if optional_features is None:
            return None, None, None

        # transform column value to a weighted feature if categorical, otherwise (int,float) keeps the weight
        def adjust_dict(user_feature_dict):
            return dict(
                (key, val) if isinstance(val, int) or isinstance(val, float) else (
                    f'{key}:{val}', CATEGORICAL_FEATURE_WEIGHT)
                for key, val in user_feature_dict.items())

        # extract all ids
        features_id = np.sort(unique(optional_features.iloc[:, USER_ID].values))
        # wrong dict without weights
        wrong_features = optional_features.set_index(optional_features.columns[USER_ID]).to_dict('index')
        # adjusted data [(id, {feature: weight,...})...],
        data_features = [(feature_id, adjust_dict(features_dict))
                         for feature_id, features_dict in wrong_features.items()]
        # extract all single features
        all_features = np.sort(unique([val for _, dicts in data_features for val in dicts.keys()]))

        return features_id, all_features, data_features


class BaseCoreModel(RexBaseModel, ABC):
    def __init__(self, provided_algorithms: [RexWrapperModel] = None):
        default_algorithms = {'KNNBaseline': KNNBaseline, 'SlopeOne': SlopeOne, 'SVD': SVD, 'LightFM': LightFM}
        self._metrics = {'recall_k', 'precision_k'}
        if provided_algorithms is not None:
            if not isinstance(provided_algorithms, dict) or \
                    not all(isinstance(algo, BaseEstimator) for _, algo in provided_algorithms):
                raise ValueError("provided algorithms must extend sklearn.base.BaseEstimator")

        self._algorithms = default_algorithms if provided_algorithms is None \
            else dict(**default_algorithms, **provided_algorithms)


class Rex(BaseCoreModel):
    def __init__(self,
                 algo: str | Iterable = 'auto',
                 auto_preprocess: bool = True,
                 metric: str | Callable[[DataFrame | PreprocessedDataFrame, RexBaseModel], float] = 'precision_k',
                 metric_params: Optional[dict] = None,
                 **kwargs):
        super(Rex, self).__init__(None)
        self._init_algo = algo
        self.models = self._compute_algorithms(algo, **kwargs)
        self.metric_params = metric_params if metric_params is not None else {}
        self.metric = self._compute_metric(metric)
        self.auto_preprocess = auto_preprocess

    def _compute_algorithms(self, algo: str | Iterable, **kwargs) -> dict[str, Any]:
        valid_algorithms = {'auto', *self._algorithms.keys()}

        if (isinstance(algo, str) and algo not in valid_algorithms) and \
                (not all(single_algo in valid_algorithms for single_algo in algo)):
            raise ValueError(f'algorithms must be one of {valid_algorithms}')

        if isinstance(algo, str) and algo != 'auto':
            return {algo: self._algorithms[algo](**kwargs)}
        else:
            used_algorithms = self._algorithms.keys() if algo == 'auto' else algo
            return {algorithm: self._algorithms[algorithm](**kwargs.get(algorithm, {}))
                    for algorithm in used_algorithms}

    def _compute_metric(self, metric):
        if isinstance(metric, str):
            def dict_to_mean(dictionary):
                return np.array(list(dictionary.values())).mean()

            if metric == 'precision_k':
                return lambda train_set, model: dict_to_mean(
                    model_evaluation.precision_k(model, train_set, **self.metric_params))

            if metric == 'recall_k':
                return lambda train_set, model: dict_to_mean(
                    model_evaluation.recall_k(model, train_set, **self.metric_params))
            else:
                raise ValueError(f"'metric' as a string must be one of the following values: {self._metrics}")
        elif isinstance(metric, Callable):
            return metric
        else:
            raise ValueError(f"'metric' must be either one of the following values: {self._metrics} "
                             f"or a (DataFrame | PreprocessedDataFrame, RexBaseModel) -> float Callable")

    def fit(self, dataset: DataFrame | PreprocessedDataFrame, y=None, verbose: bool = True, **kwargs) -> Any:
        # fit attribute, pipelines
        if self.auto_preprocess:
            self.preprocess_pipelines_ = {}
            # if it's a DataFrame -> preprocess
            if isinstance(dataset, DataFrame):
                if verbose:
                    print('Auto preprocessing weights DataFrame')
                dataset = auto_preprocess_weights_dataframe(dataset, verbose=verbose)
            # if it's a PreprocessDataFrame -> give some advice
            elif isinstance(dataset, PreprocessedDataFrame):
                dataframe_advisor(dataset.dataframe, dataset.dataframe.columns[:WEIGHT], verbose)

            # if there are user features
            if 'user_features' in kwargs:
                user_features = kwargs['user_features']
                # if user features are a DataFrame -> preprocess
                if isinstance(kwargs['user_features'], DataFrame):
                    if verbose:
                        print('Auto preprocessing user_features DataFrame')
                    kwargs['user_features'] = auto_preprocess_features_dataframe(user_features, verbose=verbose)
                # if user features are a PreprocessedDataFrame -> give some advice
                elif isinstance(user_features, PreprocessedDataFrame):
                    dataframe_advisor(user_features.dataframe,
                                      user_features.dataframe.columns[FEATURE_ID],
                                      is_feature_matrix=True,
                                      verbose=verbose)
            # if there are item features
            if 'item_features' in kwargs:
                item_features = kwargs['item_features']
                # if item features are a DataFrame -> preprocess
                if isinstance(item_features, DataFrame):
                    if verbose:
                        print('Auto preprocessing item_feature DataFrame')
                    kwargs['item_features'] = auto_preprocess_features_dataframe(item_features, verbose=verbose)
                # if item features are a PreprocessedDataFrame -> give some advice
                elif isinstance(item_features, PreprocessedDataFrame):
                    dataframe_advisor(item_features.dataframe,
                                      item_features.dataframe.columns[FEATURE_ID],
                                      is_feature_matrix=True,
                                      verbose=verbose)
            # save PreprocessPipelines
            if isinstance(dataset, PreprocessedDataFrame):
                self.preprocess_pipelines_['weights'] = PreprocessPipeline(dataset.preprocess_functions)
            if 'item_features' in kwargs and isinstance(kwargs['item_features'], PreprocessedDataFrame):
                self.preprocess_pipelines_['item_features'] = PreprocessPipeline(
                    kwargs['item_features'].preprocess_functions)
            if 'user_features' in kwargs and isinstance(kwargs['user_features'], PreprocessedDataFrame):
                self.preprocess_pipelines_['user_features'] = PreprocessPipeline(
                    kwargs['user_features'].preprocess_functions)
        else:
            # else give only advice if input data is a DataFrame
            if isinstance(dataset, DataFrame):
                dataframe_advisor(dataset, dataset.columns[:WEIGHT], verbose=verbose)
            if 'user_features' in kwargs and isinstance(kwargs['user_features'], DataFrame):
                user_features = kwargs['user_features']
                dataframe_advisor(user_features, user_features.columns[FEATURE_ID], verbose=verbose)
            if 'item_features' in kwargs and isinstance(kwargs['item_features'], DataFrame):
                item_features = kwargs['item_features']
                dataframe_advisor(item_features, item_features.columns[FEATURE_ID], verbose=verbose)

        return super(Rex, self).fit(dataset, verbose=verbose, **kwargs)

    def _checked_fit(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> Rex:
        self.scores_ = {}
        for name, model in self.models.items():
            # give advice for SurpriseModel
            if issubclass(model.__class__, SurpriseModel):
                if 'user_features' in kwargs:
                    print(f"WARNING: model '{type(model).__name__}' can't use 'user_features' in fit")
                    del kwargs['user_features']
                if 'item_features' in kwargs:
                    print(f"WARNING: model '{type(model).__name__}' can't use 'item_features' in fit")
                    del kwargs['item_features']
            # train the model
            if len(self.models) > 1:
                model.fit(dataset, verbose=verbose, **kwargs.get(name, {}))
            else:
                model.fit(dataset, verbose=verbose, **kwargs)
            self.scores_[name] = self.metric(dataset, model)
            if verbose:
                print(f"{name} score: {self.scores_[name]}")

        self.best_model_ = self.models[max(self.scores_)]
        return self

    def predict(self,
                x: Iterable | DataFrame | PreprocessedDataFrame | list[(Any, Any)],
                item_ids: Optional[Iterable] = None,
                k: int = 10,
                previous_interactions: Optional[DataFrame | PreprocessedDataFrame] = None,
                is_user_prediction: bool = True,
                **kwargs) -> DataFrame:
        # remove features and give advice
        if not isinstance(self.best_model_, LightFM):
            if 'user_features' in kwargs:
                print(f"WARNING: model '{type(self.best_model_).__name__}' can't use 'user_features' in predict")
                del kwargs['user_features']
            if 'item_features' in kwargs:
                print(f"WARNING: model '{type(self.best_model_).__name__}' can't use 'item_features' in predict")
                del kwargs['item_features']
            if 'exclude_features' in kwargs:
                print(f"WARNING: model '{type(self.best_model_).__name__}' can't exclude features in predict")
                del kwargs['exclude_features']

        if self.auto_preprocess:
            # if x is a DataFrame -> preprocess
            if isinstance(x, DataFrame):
                x = self.preprocess_pipelines_['weights'].transform(x)
            # if there are user features
            if 'user_features' in kwargs:
                user_features = kwargs['user_features']
                # if user features are a DataFrame and there is the pipeline -> preprocess
                if isinstance(user_features, DataFrame) and 'user_features' in self.preprocess_pipelines_:
                    kwargs['user_features'] = self.preprocess_pipelines_['user_features'].transform(user_features)
            # if there are item features
            if 'item_features' in kwargs:
                item_features = kwargs['item_features']
                # if user features are a DataFrame and there is the pipeline -> preprocess
                if isinstance(item_features, DataFrame) and 'item_features' in self.preprocess_pipelines_:
                    kwargs['item_features'] = self.preprocess_pipelines_['item_features'].transform(item_features)

        return super(Rex, self).predict(x,
                                        item_ids,
                                        k,
                                        previous_interactions,
                                        is_user_prediction,
                                        **kwargs.get(max(self.scores_), {}))

    def _checked_predict(self,
                         id_pairs: list[(Any, Any)],
                         k: int = 10,
                         is_user_prediction: bool = True,
                         **kwargs) -> DataFrame:
        return self.best_model_.predict(id_pairs, k=k, is_user_prediction=is_user_prediction, **kwargs)

    def get_params(self, deep=True) -> dict:
        return {model_name: model.get_params() for model_name, model in self.models.items()}

    def set_params(self, **params):
        if 'algo' in params:
            # change algorithms
            self._init_algo = params.get('algo')
            # remove algo from parameters
            del params['algo']
        self.models = self._compute_algorithms(self._init_algo, **params)
