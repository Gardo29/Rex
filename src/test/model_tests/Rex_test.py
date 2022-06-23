from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from rex.model import Rex, LightFM, SVD, KNNBaseline, RexBaseModel, SlopeOne
from rex.preprocessing2 import PreprocessPipeline, Update, Drop
from test.model_tests.rex_model_test_utility import RexModelTestUtility


class RexTest(RexModelTestUtility):
    def setUp(self) -> None:
        super(RexTest, self).setUp()
        self._dataset = self._dataset.sample(200)
        self._testset = self._dataset.sample(50)
        self._valid_algo = {'SlopeOne', 'KNNBaseline', 'LightFM', 'SVD'}

    def test_start_single_algo(self):
        for algo in self._valid_algo:
            self.not_fallible_test(lambda: Rex(algo))

    def test_ok_auto_mode(self):
        self.not_fallible_test(lambda: Rex('auto'))

    def test_error_invalid_algo(self):
        with self.assertRaises(ValueError):
            Rex('invalid_algo')

    def test_ok_initialization_with_single_algo(self):
        modifications_lightfm = {'no_components': 200, 'learning_schedule': 'adadelta'}
        self._check_attributes(Rex('LightFM', **modifications_lightfm).models['LightFM'],
                               LightFM(**modifications_lightfm),
                               modifications_lightfm)

        modifications_svd = {'n_epochs': 200, 'n_factors': 2}
        self._check_attributes(Rex('SVD', **modifications_svd).models['SVD'],
                               SVD(**modifications_svd),
                               modifications_svd)

        modifications_knn_baseline = {'min_k': 4, 'k': 200}
        self._check_attributes(Rex('KNNBaseline', **modifications_knn_baseline).models['KNNBaseline'],
                               KNNBaseline(**modifications_knn_baseline),
                               modifications_knn_baseline)

        self._check_attributes(Rex('SlopeOne').models['SlopeOne'], SlopeOne(), {})

    def test_ok_multiple_algo(self):
        for algo in self._valid_algo:
            self.not_fallible_test(lambda: Rex(self._valid_algo.difference(algo)))

    def test_ok_initialization_with_multiple_algo(self):
        mods = {
            'LightFM': {'no_components': 100, 'learning_schedule': 'adadelta'},
            'SVD': {'n_factors': 400, 'n_epochs': 200},
            'KNNBaseline': {'k': 400, 'min_k': 30}
        }
        for algo in ['auto', ['LightFM', 'SVD', 'KNNBaseline', 'SlopeOne']]:
            model = Rex(algo, **mods)
            self._check_attributes(model.models['LightFM'], LightFM(**mods['LightFM']), mods['LightFM'])
            self._check_attributes(model.models['SVD'], SVD(**mods['SVD']), mods['SVD'])
            self._check_attributes(model.models['KNNBaseline'], KNNBaseline(**mods['KNNBaseline']), mods['KNNBaseline'])
            self._check_attributes(model.models['SlopeOne'], SlopeOne(), {})

    def test_check_params_correctly(self):
        for algo in ['auto', ['LightFM', 'SVD', 'KNNBaseline', 'SlopeOne']]:
            rex_model = Rex(algo)
            self.assertTrue(all(algo in rex_model.get_params() for algo in self._valid_algo))

    def test_auto_contains_all_algos(self):
        rex = Rex('auto')
        self.assertEqual(rex.models.keys(), self._valid_algo)

    def test_set_params_single_algo(self):
        k = 400
        rex = Rex('KNNBaseline')
        rex.set_params(k=k)
        self.assertEqual(rex.models['KNNBaseline'].get_params()['k'], k)

        factors = 400
        rex.set_params(algo='SVD', n_factors=factors)
        self.assertTrue('SVD' in rex.models.keys())
        self.assertTrue(len(rex.models) == 1)
        self.assertEqual(rex.models['SVD'].get_params()['n_factors'], factors)

    def test_set_params_multiple_algo(self):
        rex = Rex('auto')
        k = 400
        factors = 400
        rex.set_params(SVD={'n_factors': 400}, KNNBaseline={'k': 400})
        self.assertEqual(rex.models['KNNBaseline'].get_params()['k'], k)
        self.assertEqual(rex.models['SVD'].get_params()['n_factors'], factors)

        learning_schedule = 'adadelta'
        rex.set_params(algo=['SlopeOne', 'LightFM'], LightFM={'learning_schedule': learning_schedule})
        self.assertEqual(rex.models['LightFM'].get_params()['learning_schedule'], learning_schedule)

    def test_fit_algo_no_errors(self):
        rex = Rex('auto')
        self.not_fallible_test(lambda: rex.fit(self._dataset))

    def test_algo_is_fitted_after_fit(self):
        rex = Rex('auto').fit(self._dataset)
        try:
            for model in rex.models.values():
                check_is_fitted(model)
        except:
            self.fail()

    def test_one_algo_is_best_algo(self):
        rex = Rex('LightFM').fit(self._dataset)
        self.assertIsInstance(rex.best_model_, LightFM)

    def test_error_wrong_dataset_no_auto_on(self):
        with self.assertRaises(ValueError):
            self._dataset['added_column'] = 0
            Rex('auto', auto_preprocess=False).fit(self._dataset)

    def test_ok_wrong_dataset_auto_on(self):
        self._dataset['added_column'] = 0
        self.not_fallible_test(lambda: self._full_rex())

    def test_auto_preprocess_pipline_saved_in_auto(self):
        self._dataset['added_column'] = 0
        rex = self._full_rex()
        self.assertTrue(all(pipeline in rex.preprocess_pipelines_
                            for pipeline in ['weights', 'item_features', 'user_features']))

    def test_empty_pipelines_no_preprocess_needed(self):
        self._dataset['rating'] = 0
        self._user_features['height'] = 0
        self._item_features['title'] = 0
        rex = Rex(auto_preprocess=True).fit(self._dataset)
        self.assertEqual(rex.preprocess_pipelines_, {})

    def test_error_wrong_format_testset_not_auto(self):
        rex = Rex('auto', auto_preprocess=False).fit(self._dataset)
        self._dataset['added_column'] = 0
        with self.assertRaises(ValueError):
            rex.predict(self._dataset.sample(100))

    def test_auto_preprocess_testset_in_auto(self):
        self._dataset['added_column'] = 0
        rex = self._full_rex()
        rex.predict(self._dataset.sample(100))

    def test_pipelines_not_present_auto_false(self):
        rex = Rex('auto', auto_preprocess=False).fit(self._dataset,
                                                     item_features=self._item_features,
                                                     user_features=self._user_features)
        self.assertFalse(hasattr(rex, 'preprocess_pipelines_'))

    def test_pipelines_not_empty_auto_true(self):
        rex = Rex('LightFM').fit(self._dataset, item_features=self._item_features, user_features=self._user_features)
        self.assertTrue(hasattr(rex, 'preprocess_pipelines_'))
        self.assertTrue(rex.preprocess_pipelines_['weights'] is not None)
        self.assertTrue(rex.preprocess_pipelines_['item_features'] is not None)
        self.assertTrue(rex.preprocess_pipelines_['user_features'] is not None)

    def test_no_pipeline_preprocessed_dataframe_input(self):
        useless_pipeline = PreprocessPipeline([
            Update({'to_remove': 1}),
            Drop('to_remove')
        ])
        self._dataset = useless_pipeline.fit_transform(self._dataset)
        self._item_features = useless_pipeline.fit_transform(self._item_features)
        self._user_features = useless_pipeline.fit_transform(self._user_features)

        rex = Rex('LightFM', auto_preprocess=False).fit(self._dataset,
                                                        item_features=self._item_features,
                                                        user_features=self._user_features)
        self.assertFalse(hasattr(rex, 'preprocess_pipelines_'))

    def check_sklearn_compatibility(self):
        self.not_fallible_test(lambda: check_estimator(self))

    def test_check_all(self):
        algos = ['auto', ['LightFM', 'SVD']]
        algo_params = [{
            'LightFM': {'no_components': 100},
            'SVD': {'n_factors': 400}
        }, {}]
        ks = [10, 5000]
        auto_preprocess = [True, False]
        metrics = ['precision_k', 'recall_k']
        metric_params = [{
            'k': 200
        }, {}]
        extra_params = [
            {'item_features': self._user_features,
             'user_features': self._item_features},
            {'item_features': self._user_features},
            {'user_features': self._item_features},
            {'LightFM': {'epochs': 50}},
            {}
        ]
        verbose = [True, False]
        is_user_predictions = [True, False]

        for algo in algos:
            for algo_param in algo_params:
                for metric_param in metric_params:
                    for preprocess in auto_preprocess:
                        for metric in metrics:
                            for is_verbose in verbose:
                                for k in ks:
                                    for is_user_prediction in is_user_predictions:
                                        for extra_param in extra_params:
                                            model = Rex(algo,
                                                        preprocess,
                                                        metric=metric,
                                                        metric_params=metric_param,
                                                        **algo_param)
                                            self.not_fallible_test(lambda: model.fit(self._dataset,
                                                                                     verbose=is_verbose,
                                                                                     **extra_param))
                                            self.not_fallible_test(lambda: model.predict(self._testset,
                                                                                         k=k,
                                                                                         is_user_prediction=is_user_prediction,
                                                                                         **extra_param))

    def _full_rex(self):
        return Rex().fit(self._dataset, item_features=self._item_features, user_features=self._user_features)

    def _check_attributes(self, model1: RexBaseModel, model2: RexBaseModel, attributes: dict):
        attr_model1 = model1.get_params()
        attr_model2 = model2.get_params()

        for k in attributes.keys():
            self.assertEqual(attr_model1[k], attr_model2[k])
