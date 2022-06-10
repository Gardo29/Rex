from rex.preprocessing import *

# Mr9cGnFAEv5u2U5
RES_PATH = '../../resources'
MOVIE_LENS_RATINGS = '/ml-small/ratings.csv'
MOVIE_LENS_MOVIES = '/ml-small/movies.csv'
MOVIE_LENS_TAGS = '/ml-small/tags.csv'

movielens = pd.read_csv(RES_PATH + MOVIE_LENS_RATINGS)
movielens_movie_features = pd.read_csv(RES_PATH + MOVIE_LENS_MOVIES)
movielens_tags = pd.read_csv(RES_PATH + MOVIE_LENS_TAGS)

movielens_preprocess = PreprocessPipeline([
    Clip('timestamp', 0, 5),
    Drop('timestamp'),
    Bin('rating', 5),
    Map('rating', lambda x: x - 1),
    Map('rating', {3: np.nan}),
    FillNa('rating', 3),
    MinMaxScaler('userId', 'rating'),

    Update(movielens['rating']),
    Normalizer('userId', 'rating'),

    Filter(lambda x: x.rating < 3),

    Update(pd.Series([1], name='LotsOfNan')),
    DropNa(),
], verbose=2)

movielens_preprocess.apply(movielens)

movielens_movie_features_preprocess = PreprocessPipeline([
    Drop('title'),
    BinThreshold('genres', 'other', 7, divider="|"),

    Update(movielens_movie_features['genres']),
    BinCumulative('genres', 'other', 50, split="|"),
    OneHotEncode('genres'),

    Select('movieId'),
    Update(movielens_movie_features['genres']),
    OneHotEncode('genres', divider="|"),
    DropDuplicates('Action')
], verbose=2)

# movielens_movie_features_preprocess.apply(movielens_movie_features)

movielens_tag_preprocess = PreprocessPipeline([
    Select(['movieId', 'tag']),
    Condense('movieId', "|"),
    OneHotEncode('tag')
], verbose=2)

# movielens_tag_preprocess.apply(movielens_tags)

"""
matrix = ToSparseMatrix('userId', 'movieId').apply(movielens.drop('timestamp', axis=1))
dropped = movielens.drop('timestamp', axis=1)
print(ToCOOMatrix('userId', 'movieId', 'rating').apply(matrix))
print(dropped)
"""
