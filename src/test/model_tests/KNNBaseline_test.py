from typing import Type

from test.model_tests.rex_model_test_utility import RexModelTestUtility
from rex.model import KNNBaseline
from surprise.prediction_algorithms import knns

from test.model_tests.surprise_test import SurpriseTest


# TODO grid search
class KNNBaselineTest(SurpriseTest):

    def _grid_search_parameters(self) -> dict:
        return {}

    def setUp(self) -> None:
        super(KNNBaselineTest, self).setUp()

    def _rex_algo(self) -> type:
        return KNNBaseline

    def _surprise_algo(self) -> type:
        return knns.KNNBaseline

    def _get_default_params(self) -> dict:
        return {'k': 40,
                'min_k': 1,
                'sim_options': {},
                'bsl_options': {},
                'verbose': True}
