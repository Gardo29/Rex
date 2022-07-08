# %% setup
import pandas as pd
import numpy as np

from rex.model import Rex
from rex.preprocessing2 import *

RES_PATH = '../../resources'
MOVIE_LENS_RATINGS = '/ml-small/ratings.csv'
MOVIE_LENS_MOVIES = '/ml-small/movies.csv'
MOVIE_LENS_TAGS = '/ml-small/tags.csv'

movielens = pd.read_csv(RES_PATH + MOVIE_LENS_RATINGS)
movielens_movie_features = pd.read_csv(RES_PATH + MOVIE_LENS_MOVIES)
movielens_tags = pd.read_csv(RES_PATH + MOVIE_LENS_TAGS)

valid_algo = {'KNNBaseline', 'SlopeOne', 'SVD', 'LightFM', 'auto'}

# rex = Rex('LightFM')
# multiple algorithms
# rex2 = Rex(['LightFM', 'KNNBaseline'], LightFM={'no_components': 20}, KNNBaseline={'k': 50, 'min_k': 2})
# single algorithm
# rex3 = Rex('SVD', n_factors=300, n_epochs=50)

# rex.fit(movielens)  # user_features=
# print(rex.predict(movielens.sample(300), item_features=movielens_movie_features, k=40, mode='item'))

# rex.fit(movielens, item_features=movielens_movie_features)  # user_features=
# print(rex.predict(movielens.sample(300), item_features=movielens_movie_features, k=40, mode='products'))

user_features_test_df = pd.DataFrame({
            'userId': np.sort(np.unique(movielens.userId.values)),
            'age': np.random.randint(10, 100, len(np.unique(movielens.userId.values))),
            'sex': np.random.randint(0, 2, len(np.unique(movielens.userId.values)))
        })
# %% Auto with item features in fit and expect that LightFM does not win
rex_auto = Rex('auto')
rex_auto.fit(movielens, item_features=movielens_movie_features)
print(rex_auto.predict(movielens.sample(200), item_features=movielens_movie_features, k=40, mode='user'))

# %% Auto and input arguments randomly and check if the winning algorithm requires them if not check if it dies
rex_random_params = Rex('auto', n_factors=200, n_epochs=50)  # SVD params
rex_random_params.fit(movielens)
print(rex_random_params.predict(movielens.sample(200), item_features=movielens_movie_features, k=40, mode='user'))

# %% Single algorithm with arguments that do not belong to it
# TODO it dies
rex_wrong_params = Rex('LightFM', n_factors=200, n_epochs=50)  # SVD params
rex_wrong_params.fit(movielens)
print(rex_wrong_params.predict(movielens.sample(200), item_features=movielens_movie_features, k=40, mode='user'))

# %% Multiple algorithms and passing just the arguments of one of them
rex_just_1_params = Rex(['LightFM', 'SVD'], LightFM={'no_components': 20})
rex_just_1_params.fit(movielens)
print(rex_just_1_params.predict(movielens.sample(200), item_features=movielens_movie_features, k=40, mode='user'))

# %% Multiple algorithms and passing the wrong arguments to an algorithm that was not declared
rex_just_1_params_wrong = Rex(['LightFM', 'SVD'], KNNBaseline={'k': 50, 'min_k': 2})
rex_just_1_params_wrong.fit(movielens)
print(rex_just_1_params_wrong.predict(movielens.sample(200), item_features=movielens_movie_features, k=40, mode='user'))

# %% Input preprocessed dataframe of dataset into rex
movielens_preprocess = Drop('timestamp').fit_transform(movielens)
rex_preprocess = Rex()
rex_preprocess.fit(movielens_preprocess)
print(rex_preprocess.predict(movielens.sample(200), item_features=movielens_movie_features, k=40, mode='user'))

# %% Preprocessed dataframe of item dataset into fit
tags_preprocessed = PreprocessPipeline([Select(['movieId', 'tag']),
                                        Condense('movieId', '|')]).fit_transform(movielens_tags)
rex_with_preproc_features = Rex('LightFM')
rex_with_preproc_features.fit(movielens, item_features=tags_preprocessed)
print(rex_with_preproc_features.predict(movielens.sample(200), item_features=tags_preprocessed,
                                        exclude_features='Comedy', k=40, mode='user'))
#%% Exclude a feature that does not exist
rex_non_existing_feat = Rex('LightFM')
rex_non_existing_feat.fit(movielens, item_features=tags_preprocessed)
print(rex_non_existing_feat.predict(movielens.sample(200), item_features=tags_preprocessed,
                                    exclude_features='Magic', k=20))
# %% Fit on a dataset that gas the same length as the predictions sample
rex_same_fit_size = Rex('auto')
rex_same_fit_size.fit(movielens.sample(200))
print(rex_same_fit_size.predict(movielens.sample(200)))

# %% Preprocessed dataframe of item dataset and user feature dataframe into fit
rex_with_user_dataset = Rex('LightFM')
rex_with_user_dataset.fit(movielens, item_features=tags_preprocessed)
print(rex_with_user_dataset.predict(movielens.sample(200), item_features=tags_preprocessed,
                                        exclude_features=['Comedy', 'Drama'], k=40, mode='user',
                                        user_features=user_features_test_df))


# %% Without rating column and making predictions on a "different" dataset
# TODO it dies
movie_col_less = movielens.drop(columns=['rating', 'timestamp'])
rex_col_less = Rex('auto')
rex_col_less.fit(movie_col_less)
print(rex_col_less.predict(movielens.sample(200)))

# %% Rating column all to 1
movielens['rating'] = 1
rex_rating_all_to_one = Rex('auto')
rex_rating_all_to_one.fit(movielens)
print(rex_rating_all_to_one.predict(movielens.sample(200), exclude_features='Other'))

# %% Change the column order to see if the auto preprocess can automatically detect the rating column (it does not)
rating_series = movielens.pop('rating')
movielens['rating'] = rating_series
rating_last_col = Rex('auto')
rating_last_col.fit(movielens)
print(rating_last_col.predict(movielens.sample(200), k=20, mode='user'))
