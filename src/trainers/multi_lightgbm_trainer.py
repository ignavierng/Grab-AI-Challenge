import logging

import base


class MultiLightGBMTrainer(base.Trainer):
    """
    Trainer class for lightgbm
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, model, dataset, params, output_dir):
        super(MultiLightGBMTrainer, self).__init__(model, dataset, params, output_dir)

    def train(self):
        # TODO: Parallelize the training (?)
        for i, regressor in enumerate(self.model.regressor_ls):
            self._logger.info('Started training for the {}-th regressor.'.format(i + 1))
            regressor.fit(self.dataset.X_train_np, self.dataset.y_train_np[:, i],
                          eval_set=[(self.dataset.X_test_np, self.dataset.y_test_np[:, i])],
                          eval_metric='rmse',
                          **self.params)
            self._logger.info('Finished training for the {}-th regressor.'.format(i + 1))

        self._logger.info('Finished training.')
        self.model.fitted = True

        # Compute test RMSE
        _, rmse = self.model.predict(self.dataset.X_test_np, self.dataset.y_test_np)
        self._logger.info('The RMSE for testing is {:.10f}'.format(rmse))

        # Save model
        model_save_path = '{}/model/'.format(self.output_dir)
        self.model.save(model_save_path)
        self._logger.info('The model is saved to {}'.format(model_save_path))
