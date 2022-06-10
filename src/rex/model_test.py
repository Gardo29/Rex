import itertools

import pandas as pd

from rex.model_selection import train_test_split
from rex.model import Rex
from rex.preprocessing import *

RES_PATH = '../../resources'
MOVIE_LENS_RATINGS = '/ml-small/ratings.csv'
MOVIE_LENS_MOVIES = '/ml-small/movies.csv'

# Best workflow for iterative training: https://github.com/lyst/lightfm/issues/194
# Lightfm: handling user and item cold-start: https://stackoverflow.com/questions/46924119/lightfm-handling-user-and-item-cold-start
# Predict using user features: https://github.com/lyst/lightfm/issues/210
# How can I learn the latent vectors of a new user?: https://github.com/lyst/lightfm/issues/371
# ValueError: https://github.com/lyst/lightfm/issues/322

movielens = pd.read_csv(RES_PATH + MOVIE_LENS_RATINGS)
movielens = Drop('timestamp').apply(movielens)

rex = Rex(algo='LightFM')
train, test = train_test_split(movielens, random_state=1234)

"""
for name, model in rex.models.items():
    print(name)
    print(model.predict(test['userId'][:3], test['movieId']))
    print()
    """
data = pd.DataFrame({
    "sas": [1, 2, 3, 4],
    "b": [6, 7, 8, 9]
})
