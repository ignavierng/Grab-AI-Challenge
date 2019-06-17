import logging
from tensorflow.keras import layers    # Needed as namespace for LSTM
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout

import base


class LSTM(base.NN):
    """
    LSTM with MLP layers (fully-connected layers) on top
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, params, _):
        super(LSTM, self).__init__(params)

    def _forward(self, inputs):
        # Stack multiple LSTM layers
        # Sandwich layer: LSTM-Dropout
        x = inputs
        for i in range(self.params['num_lstm_layers']):
            if i < self.params['num_lstm_layers'] - 1:
                x = layers.LSTM(self.params['lstm_size'], return_sequences=True)(x)
            else:
                # Final layer, do not have to return sequence
                x = layers.LSTM(self.params['lstm_size'])(x)

            if self.params['keep_prob'] < 1:
                x = Dropout(1 - self.params['keep_prob'])(x)

        # Stack multiple MLP layers
        # Sandwich layer: Sandwich layer: Dense-Batchnorm-ReLU-Dropout
        for _ in range(self.params['num_hidden_layers']):
            x = Dense(self.params['hidden_size'])(x)
            if self.params['use_bn']:
                x = BatchNormalization()(x)
            x = Activation(self.params['activation'])(x)
            if self.params['keep_prob'] < 1:
                x = Dropout(1 - self.params['keep_prob'])(x)

        outputs = Dense(5)(x)
        return outputs
