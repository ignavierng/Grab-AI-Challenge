import logging
from tensorflow.keras.layers import Convolution1D, Dense, add, Lambda

import base
from models.layers import tcn_block


class TCN(base.NN):
    """
    Temporal Convolutional Network with dilations, causal network and skip connections

    Referred from:
    - https://github.com/philipperemy/keras-tcn
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, params, _):
        super(TCN, self).__init__(params)

    def _forward(self, inputs):
        x = inputs
        x = Convolution1D(self.params['nb_filters'], 1, padding=self.params['padding'])(x)
        skip_connections = []

        # Stack multiple tcn blocks with dilations
        for s in range(self.params['nb_stacks']):
            for d in self.params['dilations']:
                x, skip_out = tcn_block(x,
                                        dilation_rate=d,
                                        nb_filters=self.params['nb_filters'],
                                        kernel_size=self.params['kernel_size'],
                                        padding=self.params['padding'],
                                        activation=self.params['activation'],
                                        keep_prob=self.params['keep_prob'],
                                        use_bn=self.params['use_bn'])
                skip_connections.append(skip_out)

        if self.params['use_skip_connections']:
            x = add(skip_connections)

        if not self.params['return_sequences']:
            x = Lambda(lambda tt: tt[:, -1, :])(x)

        outputs = Dense(5)(x)
        return outputs
