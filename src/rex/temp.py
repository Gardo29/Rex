from random import randint
import numpy as np
import pandas as pd
import surprise
from sklearn import preprocessing
from sklearn.metrics import make_scorer

import rex.model
from rex.tools import describe
from rex.model import Rex

from rex.preprocessing2 import *

RES_PATH = '../../resources'
MOVIE_LENS_RATINGS = '/ml-small/ratings.csv'
MOVIE_LENS_MOVIES = '/ml-small/movies.csv'
MOVIE_LENS_TAGS = '/ml-small/tags.csv'

movielens = pd.read_csv(RES_PATH + MOVIE_LENS_RATINGS).drop(columns='timestamp')
movielens_movie_features = pd.read_csv(RES_PATH + MOVIE_LENS_MOVIES)
movielens_tags = pd.read_csv(RES_PATH + MOVIE_LENS_TAGS)

valid_algo = {'KNNBaseline', 'SlopeOne', 'SVD', 'LightFM', 'auto'}

rex = Rex('LightFM', auto_preprocess=False)
rex2 = Rex(['SVD', 'LightFM'], SVD={'n_factors': 300, 'n_epochs': 50}, LightFM={'no_components': 50})

rex.fit(movielens, item_features=movielens_movie_features)  # user_features=
print(rex.predict(movielens.sample(300), item_features=movielens_movie_features, k=40, mode='item'))
