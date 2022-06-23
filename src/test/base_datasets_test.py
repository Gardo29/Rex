from unittest import TestCase
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
import numpy as np

from rex.preprocessing2 import Map, PreprocessPipeline, DropDuplicates, Drop
from rex.tools import unique
from test.base_test import BaseTest

RES_PATH = '../../../resources'
MOVIE_LENS_RATINGS = '/ml-small/ratings.csv'
MOVIE_LENS_MOVIES = '/ml-small/movies.csv'
SEPARATOR = ','


class BaseDatasetsTest(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        self._dataset = pd.read_csv(RES_PATH + MOVIE_LENS_RATINGS, sep=SEPARATOR)
        self._item_features = pd.read_csv(RES_PATH + MOVIE_LENS_MOVIES, sep=SEPARATOR)
        self._user_features = pd.DataFrame({
            'userId': np.sort(unique(self._dataset.userId.values)),
            'age': np.random.randint(10, 100, len(unique(self._dataset.userId.values)))
        })
        self._random_state = 1234
        # replace movie id with titles
        self._dataset = PreprocessPipeline([
            Map('movieId', {movie_id: name for movie_id, name in self._item_features.iloc[:, :2].values}),
            DropDuplicates(['userId', 'movieId'])
        ], verbose=False).fit_transform(self._dataset).dataframe
        self._item_features = PreprocessPipeline([
            Drop('movieId'),
            DropDuplicates(subset_features='title')], verbose=False).fit_transform(self._item_features).dataframe
