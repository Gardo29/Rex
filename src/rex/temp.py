from __future__ import annotations

import io
from typing import Optional, List, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame
import json

from rex.api import remote_preprocess, remote_fit, remote_predict, remote_fit_predict

from rex.preprocessing import Bin, PreprocessFunction, Drop

RES_PATH = '../../resources'
MOVIE_LENS_RATINGS = '/ml-small/ratings.csv'
MOVIE_LENS_MOVIES = '/ml-small/movies.csv'
MOVIE_LENS_TAGS = '/ml-small/tags.csv'

movielens = pd.read_csv(RES_PATH + MOVIE_LENS_RATINGS).drop(columns='timestamp')
movielens_movie_features = pd.read_csv(RES_PATH + MOVIE_LENS_MOVIES)
movielens_tags = pd.read_csv(RES_PATH + MOVIE_LENS_TAGS)

movielens_movie_features['real'] = np.random.random(len(movielens_movie_features)) * 5

print(remote_preprocess("../resources" + MOVIE_LENS_RATINGS, "preprocess_dataframe3.csv",
                        [Bin('rating', 5), Drop('timestamp')]))

print(remote_fit('d.csv', 'lfmmodel', {'algo': 'LightFM'}))
print(remote_predict('lfmmodel', 'predictions.csv', np.unique(movielens.userId)[:20], movielens.movieId[:5]))

print(remote_fit_predict("ratings.csv", "model_fit_predict", "predictions",
                         np.unique(movielens.userId)[:20], movielens.movieId[:5], model_parameters={'algo': 'LightFM'}))
