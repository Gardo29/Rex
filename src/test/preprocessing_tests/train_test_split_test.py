from pandas import DataFrame

from rex import model_selection
from rex.model_selection import generate_train_test, train_anti_test_split
from rex.preprocessing import PreprocessPipeline, Update, Drop, PreprocessedDataFrame
from test.base_datasets_test import BaseDatasetsTest


class TrainTestSplitTest(BaseDatasetsTest):
    def setUp(self) -> None:
        super(TrainTestSplitTest, self).setUp()
        self._dataset = PreprocessPipeline([
            Drop('timestamp')
        ]).fit_transform(self._dataset).dataframe
        self._preprocessed_dataset = PreprocessPipeline([
            Update({"new_col": 1}),
            Drop('new_col')
        ]).fit_transform(self._dataset)

    def test_train_test_split_dataframe(self):
        train, test = generate_train_test(self._dataset)
        self.assertIsInstance(train, DataFrame)
        self.assertIsInstance(test, DataFrame)
        self.assertEqual(train, self._dataset)

    def test_train_test_split_preprocessed_dataframe(self):
        train, test = generate_train_test(self._preprocessed_dataset)
        self.assertIsInstance(train, PreprocessedDataFrame)
        self.assertIsInstance(test, PreprocessedDataFrame)
        self.assertEqual(train, self._preprocessed_dataset)

    def test_train_test_split_percent(self):
        percent = 0.7
        train, test = generate_train_test(self._dataset, train_size=percent)
        self.assertEqual(len(test), round(len(train) * percent))

    def test_train_test_split_number(self):
        n_instances = 200
        train, test = generate_train_test(self._dataset, train_size=n_instances)
        self.assertEqual(len(test), n_instances)

    def test_train_anti_test_dataframe(self):
        train, test = train_anti_test_split(self._dataset)
        self.assertIsInstance(train, DataFrame)
        self.assertIsInstance(test, DataFrame)
        self.assertEqual(train, self._dataset)
        train_pairs = {(uid, iid) for uid, iid in train.values[:, :2]}
        test_pairs = {(uid, iid) for uid, iid in test.values[:, :2]}
        self.assertEqual(train_pairs & test_pairs, set())

    def test_train_anti_test_preprocessed_dataframe(self):
        train, test = train_anti_test_split(self._preprocessed_dataset)
        self.assertIsInstance(train, PreprocessedDataFrame)
        self.assertIsInstance(test, PreprocessedDataFrame)
        self.assertEqual(train, self._preprocessed_dataset)
        train_pairs = {(uid, iid) for uid, iid in train.dataframe.values[:, :2]}
        test_pairs = {(uid, iid) for uid, iid in test.dataframe.values[:, :2]}
        self.assertEqual(train_pairs & test_pairs, set())
