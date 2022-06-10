from __future__ import annotations

import re
from typing import (Union, Optional, Any, Callable, Iterable, TypeVar)
import itertools
from abc import abstractmethod, ABC
from collections import defaultdict
from itertools import (product, groupby)  # for surprise predict
from time import time
import threading
import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from rex.model_evaluation import precision_k
from rex.preprocessing2 import PreprocessedDataFrame, auto_preprocess_weights_dataframe, PreprocessPipeline, \
    auto_preprocess_features_dataframe
from rex.tools import check_weights, check_features, unique, dataframe_advisor
# LightFM
from lightfm import data
import lightfm
# surprise
import surprise
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise import Reader, Prediction
from surprise import prediction_algorithms

USER_ID = 0
ITEM_ID = 1
FEATURE_ID = 0
WEIGHT = 2


# ---------------- UTILITY FUNCTIONS --------------
def compute_predictions(predictions,
                        k: int,
                        key: Callable,
                        map_to: Callable,
                        sort_by: Callable,
                        filter_by: Optional[Callable] = None,
                        extract_ids: Optional[Callable] = None,
                        previous_interaction: Optional[DataFrame] = None) -> dict:
    def select_predictions(grouped_predictions):
        # if previous_interactions are present remove them from the prediction
        if previous_interaction is not None and extract_ids is not None:
            grouped_predictions = filter(
                lambda x: extract_ids(x) not in previous_interaction.iloc[:, [USER_ID, ITEM_ID]].values.tolist(),
                grouped_predictions
            )
        # in case filter by function
        if filter_by:
            grouped_predictions = filter(filter_by, grouped_predictions)
        # sort predictions by ranking
        sorted_predictions = sorted(grouped_predictions, key=sort_by, reverse=True)
        # select first k
        k_predictions = sorted_predictions[:k]
        # extract only ids
        mapped_predictions = map(map_to, k_predictions)
        return list(mapped_predictions)

    # group by key, and extract for each id top k predictions
    return {key_id: select_predictions(grouped_predictions) for key_id, grouped_predictions in
            groupby(sorted(predictions, key=key), key=key)}


# ---------------- MODELS --------------

class RexBaseModel(ABC, BaseEstimator):
    def fit(self, dataset: DataFrame | PreprocessedDataFrame, y=None, verbose: bool = True, **kwargs) -> Any:
        check_weights(dataset)
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
        # fit the model
        self._checked_fit(dataset, verbose, **kwargs)
        if verbose:
            print(f"Fit time '{self.__class__.__name__}': {time() - start_time}s")
        return self

    @abstractmethod
    def _checked_fit(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> Any:
        pass

    def predict(self,
                x: Iterable | DataFrame | PreprocessedDataFrame,
                item_ids: Optional[Iterable] = None,
                k: int = 10,
                previous_interactions: Optional[DataFrame] = None,
                mode: str = "user",
                **kwargs) -> DataFrame:
        # check if model is fitted
        check_is_fitted(self)

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
            if (mode == 'user' and 'item_features' not in kwargs) or (mode != 'user' and 'user_features' not in kwargs):
                raise ValueError("If 'exclude_features' is specified relative features must be provided")
            kwargs['exclude_features'] = list(np.atleast_1d(kwargs['exclude_features']))

        # if it's PreprocessedDataFrame extract
        if isinstance(x, PreprocessedDataFrame):
            x = x.dataframe
        # if it's a normal DataFrame check DataFrame and extract ids
        if isinstance(x, DataFrame):
            check_weights(x)
            user_ids = np.sort(unique(x.iloc[:, USER_ID].values))
            item_ids = np.sort(unique(x.iloc[:, ITEM_ID].values))
        # if there are ids pairs all work is done, return
        elif isinstance(x, Iterable) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in x):
            return self._checked_predict(list(x), k, previous_interactions, mode, **kwargs)
        # if there are two ids iterable just pass them
        elif isinstance(x, Iterable) and isinstance(item_ids, Iterable):
            user_ids = np.sort(unique(x))
            item_ids = np.sort(unique(item_ids))
        else:
            raise ValueError(
                "'x' must be either a Pandas Dataframe or "
                "provide both 'x' and 'item_id' as Iterables or "
                "provide a unique Iterable with ids couples")
        # run the predictions
        pairs = list(product(user_ids, item_ids))
        return self._checked_predict(pairs, k, previous_interactions, mode, **kwargs)

    @abstractmethod
    def _checked_predict(self,
                         ids_pairs: list[(Any, Any)],
                         k: int = 10,
                         previous_interactions: Optional[DataFrame] = None,
                         mode: str = "user",
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
        super(RexWrapperModel, self).set_params()
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
                         ids_pairs: list[(Any, Any)],
                         k: int = 10,
                         previous_interactions: Optional[DataFrame] = None,
                         mode: str = "user",
                         **kwargs) -> DataFrame:
        # calculates all predictions
        all_predictions = [self._model.predict(user_id, item_id, clip=False) for user_id, item_id in ids_pairs]
        # extract top k predictions
        k_top_predictions = compute_predictions(
            predictions=all_predictions,
            previous_interaction=previous_interactions,
            extract_ids=lambda x: [x.uid, x.iid],
            filter_by=lambda x: not x.details['was_impossible'],
            k=k,
            key=lambda x: x.uid if mode == 'user' else x.iid,
            map_to=lambda x: x.iid if mode == 'user' else x.uid,
            sort_by=lambda x: x.est
        )
        # create dataframe from predictions dict
        return DataFrame(k_top_predictions)


class SVD(SurpriseModel):
    def __init__(self, **kwargs):
        # TODO: rimuovi
        kwargs['n_epochs'] = 1
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
        if verbose:
            self._model = self._model_class(verbose=True, **self._params)
        self._model.fit(dataset)
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


# TODO: filtering in base alle classi
class LightFM(RexWrapperModel):
    def __init__(self, **kwargs):
        super(LightFM, self).__init__(lightfm.LightFM, **kwargs)
        self._categorical_weight = 1

    def _checked_fit(self, dataset: DataFrame, verbose: bool = True, **kwargs) -> LightFM:
        # extract features data: ids, all possible features, [id,{feature:weight,...}..]
        features_user_ids, all_user_features, user_features = self._extract_features_data(
            kwargs.get('user_features', None))
        features_item_ids, all_item_features, item_features = self._extract_features_data(
            kwargs.get('item_features', None))
        # TODO: meglio qui o nel costruttore?
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
        self._model.fit(interaction_matrix,
                        sample_weight=weights,
                        num_threads=threading.active_count(),  # TODO: remove?
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
                         ids_pairs: list[(Any, Any)],
                         k: int = 10,
                         previous_interactions: Optional[DataFrame] = None,
                         mode: str = "user",
                         **kwargs) -> DataFrame:

        # if there are features to be excluded
        if 'exclude_features' in kwargs:
            exclude_features = kwargs['exclude_features']
            # remove them from kwargs
            del kwargs['exclude_features']
            # check consistency
        else:
            exclude_features = None
        # rebuild features matrices and eventually filter for complementary
        if kwargs and 'user_features' in kwargs:
            user_ids, all_user_features, user_features = self._extract_features_data(kwargs.get('user_features'))
            self._lfm_builder.fit_partial(users=user_ids, user_features=all_user_features)
            kwargs['user_features'] = self._lfm_builder.build_user_features(user_features)
            # if there are features to be excluded and mode isn't 'user'
            if exclude_features and mode != 'user':
                # compute users to be excluded
                user_to_be_excluded = [user_id
                                       for user_id, features in user_features
                                       if not any(feature in exclude_features for feature in features.keys())]
                # remove those users from ids_pairs
                ids_pairs = [(user_id, item_id) for user_id, item_id in ids_pairs if user_id not in user_to_be_excluded]

        if kwargs and 'item_features' in kwargs:
            item_ids, all_item_features, item_features = self._extract_features_data(kwargs.get('item_features'))
            self._lfm_builder.fit_partial(items=item_ids, item_features=all_item_features)
            kwargs['item_features'] = self._lfm_builder.build_item_features(item_features)
            # if there are features to be excluded and mode is 'user'
            if exclude_features and mode == 'user':
                # compute users to be excluded
                item_to_be_excluded = [item_id
                                       for item_id, features in item_features
                                       if any(feature in exclude_features for feature in features.keys())]
                # remove those users from ids_pairs
                ids_pairs = [(user_id, item_id) for user_id, item_id in ids_pairs if item_id not in item_to_be_excluded]

        # map user ids
        lightfm_user_id = np.array([self._user_id_map[user_id] for user_id, _ in ids_pairs])
        unchanged_user_id = [user_id for user_id, _ in ids_pairs]
        # map items ids
        lightfm_item_id = np.array([self._item_id_map[item_id] for _, item_id in ids_pairs])
        unchanged_item_id = [item_id for _, item_id in ids_pairs]

        # compute scores
        scores = self._model.predict(lightfm_user_id, lightfm_item_id, num_threads=threading.active_count(), **kwargs)
        # compute top k predictions
        k_predictions = compute_predictions(
            predictions=list(zip(unchanged_user_id, unchanged_item_id, scores)),
            extract_ids=lambda x: [x[USER_ID], x[ITEM_ID]],
            k=k,
            key=lambda x: x[USER_ID] if mode == 'user' else x[ITEM_ID],
            map_to=lambda x: x[ITEM_ID] if mode == 'user' else x[USER_ID],
            sort_by=lambda x: x[WEIGHT]
        )
        return DataFrame(k_predictions)

    @staticmethod
    def _concat_unique_list(list1: np.ndarray, list2: Optional[np.ndarray]) -> np.ndarray:
        if list2 is None:
            return np.sort(unique(list1))
        else:
            return np.sort(unique(np.append(list1, list2)))

    def _extract_features_data(self, optional_features: Optional[DataFrame]) -> (Optional[np.ndarray],
                                                                                 Optional[set],
                                                                                 Optional[list[tuple[Any, dict]]]):
        if optional_features is None:
            return None, None, None

        # transform column value to a weighted feature if categorical, otherwise (int,float) keeps the weight
        def adjust_dict(user_feature_dict):
            return dict(
                (key, val) if isinstance(val, int) or isinstance(val, float) else (val, self._categorical_weight)
                for key, val in user_feature_dict.items())

        # extract all ids
        features_id = np.sort(unique(optional_features.iloc[:, USER_ID].values))
        # wrong dict without weights
        wrong_features = optional_features.set_index(optional_features.columns[USER_ID]).to_dict('index')
        # adjusted data [(id, {feature: weight,...})...],
        data_features = np.sort([(feature_id, adjust_dict(features_dict))
                                 for feature_id, features_dict in wrong_features.items()])
        # extract all single features
        all_features = np.sort(unique([val for _, dicts in data_features for val in dicts.keys()]))

        return features_id, all_features, data_features


class BaseCoreModel(RexBaseModel, ABC):
    def __init__(self, provided_algorithms: [RexWrapperModel] = None):
        default_algorithms = {'KNNBaseline': KNNBaseline, 'SlopeOne': SlopeOne, 'SVD': SVD, 'LightFM': LightFM}

        if provided_algorithms is not None:
            if not isinstance(provided_algorithms, dict) or \
                    not all(isinstance(algo, BaseEstimator) for _, algo in provided_algorithms):
                raise ValueError("provided algorithms must extend sklearn.base.BaseEstimator")

        self._algorithms = default_algorithms if provided_algorithms is None \
            else dict(**default_algorithms, **provided_algorithms)


class Rex(BaseCoreModel):
    def __init__(self, algo: str | Iterable = 'auto', auto_preprocess=True, metric='precision_k', **kwargs):
        super(Rex, self).__init__(None)
        self._init_algo = algo
        self.models = self._compute_algorithms(algo, **kwargs)
        self.metric = metric
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

        # eventually save PreprocessPipelines
        if isinstance(dataset, PreprocessedDataFrame):
            self.preprocess_pipelines_['weights'] = PreprocessPipeline(dataset.preprocess_functions)
        if 'item_features' in kwargs and isinstance(kwargs['item_features'], PreprocessedDataFrame):
            self.preprocess_pipelines_['item_features'] = PreprocessPipeline(
                kwargs['item_features'].preprocess_functions)
        if 'user_features' in kwargs and isinstance(kwargs['user_features'], PreprocessedDataFrame):
            self.preprocess_pipelines_['user_features'] = PreprocessPipeline(
                kwargs['user_features'].preprocess_functions)
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
            self.scores_[name] = np.random.randint(0, 10)  # TODO evaluate model

        self.best_model_ = self.models[max(self.scores_)]
        return self

    def predict(self,
                x: Iterable | DataFrame | PreprocessedDataFrame | list[(Any, Any)],
                item_ids: Optional[Iterable] = None,
                k: int = 10,
                previous_interactions: Optional[DataFrame | PreprocessedDataFrame] = None,
                mode: str = "user",
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

        return super(Rex, self).predict(x, item_ids, k, previous_interactions, mode, **kwargs)

    def _checked_predict(self,
                         ids_pairs: list[(Any, Any)],
                         k: int = 10,
                         previous_interactions: Optional[DataFrame] = None,
                         mode: str = "user",
                         **kwargs) -> DataFrame:
        return self.best_model_.predict(ids_pairs, None, k, previous_interactions, mode, **kwargs)

    def get_params(self, deep=True) -> dict:
        return {model_name: model.get_params() for model_name, model in self.models.items()}

    def set_params(self, **params):
        if 'algo' in params:
            # change algorithms
            self._init_algo = params.get('algo')
            # remove algo from parameters
            del params['algo']
        self.models = self._compute_algorithms(self._init_algo, **params)
