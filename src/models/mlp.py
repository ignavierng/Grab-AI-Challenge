import logging
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout

import base


class MLP(base.NN):
    """
    Referred from:
    - https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
    - https://towardsdatascience.com/intuit-and-implement-batch-normalization-c05480333c5b
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, params, _):
        super(MLP, self).__init__(params)

    def _forward(self, inputs):
        # Stack multiple MLP layers
        # Sandwich layer: Dense-Batchnorm-ReLU-Dropout
        x = inputs
        for _ in range(self.params['num_hidden_layers']):
            x = Dense(self.params['hidden_size'])(x)
            if self.params['use_bn']:
                x = BatchNormalization()(x)
            x = Activation(self.params['activation'])(x)
            if self.params['keep_prob'] < 1:
                x = Dropout(1 - self.params['keep_prob'])(x)

        outputs = Dense(5)(x)
        return outputs
