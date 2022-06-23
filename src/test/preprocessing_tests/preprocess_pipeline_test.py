import pandas as pd

from rex.preprocessing2 import PreprocessPipeline, Map, Drop, Bin
from test.base_test import BaseTest
from test.base_datasets_test import BaseDatasetsTest


class PreprocessPipelineTest(BaseTest):
    def setUp(self) -> None:
        super(PreprocessPipelineTest, self).setUp()
        self._data = pd.DataFrame({
            'timestamp': range(5),
            'rating': [1.2, 3, 2.4, 4.5, 5]
        })

    def test_ok_no_functions(self):
        self.not_fallible_test(lambda: PreprocessPipeline([]))

    def test_error_not_iterable(self):
        with self.assertRaises(ValueError):
            PreprocessPipeline(Map('timestamp'))

    def test_ok_multiple_functions(self):
        self.not_fallible_test(lambda: PreprocessPipeline([Drop('timestamp'), Bin('rating', 5)]))

    def test_ok_functions_are_consecutive(self):
        bins = 5
        pipeline_result = PreprocessPipeline([
            Drop('timestamp'),
            Bin('rating', bins)
        ]).fit_transform(self._data)
        manual_drop = Drop('timestamp').fit_transform(self._data)
        manual_result = Bin('rating', bins).fit_transform(manual_drop)

        self.assertEqual(pipeline_result, manual_result)
        self.assertEqual(pipeline_result.dataframe, manual_result.dataframe)
        self.assertEqual(pipeline_result.preprocess_functions, manual_result.preprocess_functions)
        self.assertEqual(PreprocessPipeline(pipeline_result.preprocess_functions),
                         PreprocessPipeline(manual_result.preprocess_functions))
