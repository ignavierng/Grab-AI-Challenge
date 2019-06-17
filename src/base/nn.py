import os
import numpy as np
from math import sqrt
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import backend as K

from base.model import Model
from helpers.dir_utils import create_dir
from helpers.tf_utils import is_cuda_available, tensor_description


class NN(Model, ABC):
    """
    Abstract base class for neural network (Tensorflow)
    """

    def __init__(self, params):
        super(NN, self).__init__(params)

        self._build()
        self._init_session()
        self._init_saver()

    @abstractmethod
    def _forward(self, inputs):
        pass

    def _build(self):
        """
        Build Tensorflow graph
        """
        tf.reset_default_graph()

        # Initialize placeholders
        self.lr = tf.placeholder(tf.float32, None, name='lr')

        if self.params['use_sliding_inputs']:
            self.inputs = tf.placeholder(tf.float32,
                                         [None, self.params['num_steps'], self.params['input_size']],
                                         name='inputs')
        else:
            self.inputs = tf.placeholder(tf.float32, [None, self.params['input_size']], name='inputs')

        self.targets = tf.placeholder(tf.float32, [None, 5], name='targets')

        # Forward pass
        self.outputs = self._forward(self.inputs)

        # For debugging
        self._logger.debug('inputs.shape'.format(self.inputs.shape))
        self._logger.debug('targets.shape'.format(self.targets.shape))
        self._logger.debug('outputs.shape'.format(self.outputs.shape))

        # Loss and optimizer
        l2_reg_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.mse = tf.reduce_mean(tf.square(self.outputs - self.targets), name='loss_mse')

        self.loss = self.mse + self.params['l2_reg'] * l2_reg_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name='train_op')

        self._logger.debug('Finished building graph')

    def predict(self, X_np, y_np):
        """
        - Shape of X_np:
              - if sliding features are used: [num_batch, batch_size, num_steps, num_features]
              - if sliding features are not used: [num_batch, batch_size, num_features]
        - Shape of y_np: [num_batch, batch_size, 5]
        - Return:
            - y_pred_np
            - rmse
        """
        pred_ls = []
        mse_ls = []
        for batch_step, (X_batch_np, y_batch_np) in enumerate(zip(X_np, y_np)):
            batch_step += 1    # Start from batch=1
            curr_pred, curr_mse = self.sess.run(
                    [self.outputs, self.mse], feed_dict={
                        self.inputs: X_batch_np,
                        self.targets: y_batch_np,
                        K.learning_phase(): 0
                    }
            )
            pred_ls.append(curr_pred)
            mse_ls.append(curr_mse)
        # Calculate rmse by averaging mse across all batches then compute its sqrt
        return np.array(pred_ls), sqrt(np.mean(mse_ls))

    def _init_session(self):
        if is_cuda_available() and self.params['use_cuda']:
            # Use GPU
            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.5,
                    allow_growth=True,
                )
            ))
        else:
            self.sess = tf.Session()

    def _init_saver(self):
        self.saver = tf.train.Saver()

    def save(self, file_path):
        # Example of file_path: './model/'
        create_dir(file_path)
        self.saver.save(self.sess, file_path)

    def load(self, metafile_path):
        # Example of metafile_path: 'output/2019-06-16_15-26-36-656/model/epoch_3/model.meta'
        if not os.path.exists(metafile_path):
            raise ValueError('model_file does not exist at {}'.format(metafile_path))
        self.saver = tf.train.import_meta_graph(metafile_path)
        deepest_dir = metafile_path[:-len(os.path.basename(metafile_path))]
        self.saver.restore(self.sess, tf.train.latest_checkpoint(deepest_dir))
        
        self.fitted = True

    def print_summary(self, print_func):
        """
        Print a summary table of the network structure

        Referred from:
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/model_analyzer.py
        """
        variables = tf.trainable_variables()

        print_func('Model summary:')
        print_func('---------')
        print_func('Variables: name (type shape) [size]')
        print_func('---------')

        total_size = 0
        total_bytes = 0
        for var in variables:
            # if var.num_elements() is None or [] assume size 0.
            var_size = var.get_shape().num_elements() or 0
            var_bytes = var_size * var.dtype.size
            total_size += var_size
            total_bytes += var_bytes

            print_func('{} {} [{}, bytes: {}]'.format(var.name, tensor_description(var), var_size, var_bytes))

        print_func('Total size of variables: {}'.format(total_size))
        print_func('Total bytes of variables: {}'.format(total_bytes))
