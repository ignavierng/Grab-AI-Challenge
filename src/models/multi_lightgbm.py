import logging
from math import sqrt
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

import base
from helpers.dir_utils import create_dir


class MultiLightGBM(base.Model):
    _logger = logging.getLogger(__name__)

    def __init__(self, params, _):
        super(MultiLightGBM, self).__init__(params)
        self._build()

    def _build(self):
        self.regressor_ls = [lgb.LGBMRegressor(**self.params) for _ in range(5)]
        self._logger.info('Finished building model.')

    def predict(self, X_np, y_np):
        pred_np = np.array([regressor.predict(X_np) for regressor in self.regressor_ls]).swapaxes(1, 0)
        rmse = sqrt(mean_squared_error(y_np, pred_np))
        return pred_np, rmse

    def save(self, folder_path):
        create_dir(folder_path)
        for i, regressor in enumerate(self.regressor_ls):
            joblib.dump(regressor, '{}/lgb_{}.pkl'.format(folder_path, i + 1))

    def load(self, folder_path):
        for i in range(5):
            self.regressor_ls[i] = joblib.load('{}/lgb_{}.pkl'.format(folder_path, i + 1))

        self.fitted = True

    def print_summary(self, print_func):
        print_func('Model summary:')
        for i, regressor in enumerate(self.regressor_ls):
            print_func('{}-th regressor: \n{}'.format(i + 1, regressor))
