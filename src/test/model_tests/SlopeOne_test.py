from surprise.prediction_algorithms import knns, slope_one

from test.model_tests.rex_model_test_utility import RexModelTestUtility
from rex.model import SlopeOne
from test.model_tests.surprise_test import SurpriseTest


# TODO grid search
class SlopeOneTest(SurpriseTest):
    def setUp(self) -> None:
        super(SlopeOneTest, self).setUp()

    def _grid_search_parameters(self) -> dict:
        return {}

    def _rex_algo(self):
        return SlopeOne

    def _surprise_algo(self):
        return slope_one.SlopeOne

    def _get_default_params(self) -> dict:
        return dict()

    def test_fit_no_weights(self):
        return super(SlopeOneTest, self).test_fit_no_weights()

    def test_predict(self):
        self.check_equals_predict(random_state=False)
