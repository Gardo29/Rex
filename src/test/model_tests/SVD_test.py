import surprise

from rex.model import SVD
from rex_model_test_utility import RexModelTestUtility
from test.model_tests.surprise_test import SurpriseTest


# TODO grid search
class SVDTest(SurpriseTest):

    def setUp(self) -> None:
        super(SVDTest, self).setUp()

    def _grid_search_parameters(self) -> dict:
        return {}

    def _rex_algo(self) -> type:
        return SVD

    def _surprise_algo(self) -> type:
        return surprise.SVD

    def _get_default_params(self) -> dict:
        return {'n_factors': 100,
                'n_epochs': 20,
                'biased': True,
                'init_mean': 0,
                'init_std_dev': 0.1,
                'lr_all': 0.005,
                'reg_all': 0.02,
                'lr_bu': None,
                'lr_bi': None,
                'lr_pu': None,
                'lr_qi': None,
                'reg_bu': None,
                'reg_bi': None,
                'reg_pu': None,
                'reg_qi': None,
                'random_state': None,
                'verbose': False}

    def test_predict(self):
        self.check_equals_predict(random_state=True)
