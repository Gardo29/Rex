import pandas as pd

from rex.model import Rex

RES_PATH = '../../resources'
MOVIE_LENS_RATINGS = '/ml-small/ratings.csv'
MOVIE_LENS_MOVIES = '/ml-small/movies.csv'
MOVIE_LENS_TAGS = '/ml-small/tags.csv'

movielens = pd.read_csv(RES_PATH + MOVIE_LENS_RATINGS).drop(columns='timestamp')
movielens_movie_features = pd.read_csv(RES_PATH + MOVIE_LENS_MOVIES)
movielens_tags = pd.read_csv(RES_PATH + MOVIE_LENS_TAGS)

valid_algo = {'KNNBaseline', 'SlopeOne', 'SVD', 'LightFM', 'auto'}

rex = Rex('LightFM')

# rex.fit(movielens)  # user_features=
# print(rex.predict(movielens.sample(300), item_features=movielens_movie_features, k=40, mode='item'))

rex.fit(movielens, item_features=movielens_movie_features)  # user_features=
print(rex.predict(movielens.sample(300), item_features=movielens_movie_features, k=40, mode='item'))
