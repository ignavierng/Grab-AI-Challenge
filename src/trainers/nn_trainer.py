import logging
import numpy as np
from math import sqrt
import tensorflow as tf
from tensorflow.keras import backend as K

import base


class NNTrainer(base.Trainer):
    """
    Trainer class for neural network
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, model, dataset, params, output_dir):
        super(NNTrainer, self).__init__(model, dataset, params, output_dir)
        self.sess = model.sess

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        stop_epochs = 0    # For early stopping

        # Store statistics of best epoch
        best_stats_dict = {}
        best_stats_dict['epoch'] = 0
        best_stats_dict['val_rmse'] = np.inf
        best_stats_dict['test_rmse'] = np.inf

        self._logger.info('Start training for {} epochs'.format(self.params['epochs']))

        # Log the rmse before training started
        for mode, X_np, y_np in [('train', self.dataset.X_train_np, self.dataset.y_train_np),
                                 ('val', self.dataset.X_val_np, self.dataset.y_val_np),
                                 ('test', self.dataset.X_test_np, self.dataset.y_test_np)]:
            _, rmse = self.model.predict(X_np, y_np)
            self._logger.info('[{}] [Epoch: {}] rmse: {:.10f}'.format(mode, 0, rmse))

        for epoch in range(1, self.params['epochs'] + 1):
            if stop_epochs >= self.params['early_stopping_rounds']:
                self._logger.info('[update] [Epoch: {}] Early stopping'.format(epoch))
                break

            stop_epochs, best_stats_dict = self.train_epoch(epoch, stop_epochs, best_stats_dict)

        # Finished training
        self.model.fitted = True
        self._logger.info('Finished training for {} epochs'.format(epoch))
        self._logger.info('Best model is obtained at {}-th epoch and test rmse is {:.10f}'.format(
            best_stats_dict['epoch'], best_stats_dict['test_rmse']
        ))

    def train_epoch(self, epoch, stop_epochs, best_stats_dict):
        decayed_lr = self.params['lr'] * (self.params['lr_decay'] ** max(float(epoch + 1), 0.0))

        ##### Training #####
        for batch_step, (X_batch_np, y_batch_np) in enumerate(zip(self.dataset.X_train_np,
                                                                  self.dataset.y_train_np)):
            batch_step += 1    # Start from batch=1
            train_mse, _ = self.sess.run(
                    [self.model.mse, self.model.train_op], feed_dict={
                        self.model.lr: decayed_lr,
                        self.model.inputs: X_batch_np,
                        self.model.targets: y_batch_np,
                        K.learning_phase(): 1
                    }
            )
            self._logger.info('[train] [Epoch: {}] [Batch: {}] [Learning rate: {:.4E}] rmse: {:.10f}'.format(
                epoch, batch_step, decayed_lr, sqrt(train_mse)
            ))

        ##### Validation #####
        _, val_rmse = self.model.predict(self.dataset.X_val_np, self.dataset.y_val_np)
        self._logger.info('[val] [Epoch: {}] rmse: {:.10f}'.format(epoch, val_rmse))

        ##### Testing #####
        _, test_rmse = self.model.predict(self.dataset.X_test_np, self.dataset.y_test_np)
        self._logger.info('[test] [Epoch: {}] rmse: {:.10f}'.format(epoch, test_rmse))

        ##### Best model is found (based on val_rmse) #####
        if val_rmse < best_stats_dict['val_rmse']:
            self._logger.info('[update] [Epoch: {}] best_val_rmse updated from {:.10f} to {:.10f}'.format(
                epoch, best_stats_dict['val_rmse'], val_rmse
            ))

            # Store statistics of best model
            stop_epochs = 0
            best_stats_dict['epoch'] = epoch
            best_stats_dict['val_rmse'] = val_rmse
            best_stats_dict['test_rmse'] = test_rmse

            if self.params['save_best_model']:
                # Save the best model
                model_save_path = '{}/model/epoch_{}/model'.format(self.output_dir, epoch)
                self._logger.info('[update] [Epoch: {}] current best model saved to {}'.format(epoch, model_save_path))
                self.model.save(file_path=model_save_path)
        else:
            stop_epochs += 1

        return stop_epochs, best_stats_dict
