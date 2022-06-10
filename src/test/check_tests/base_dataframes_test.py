from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from test.base_datasets_test import BaseDatasetsTest
import numpy as np


class BaseCheckTest(BaseDatasetsTest, ABC):
    def _exception_test(self, dataframe):
        with self.assertRaises(ValueError):
            self._function()(dataframe)

    @abstractmethod
    def _function(self) -> Callable:
        pass
