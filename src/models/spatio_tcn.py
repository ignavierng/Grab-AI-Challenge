import logging
from tensorflow.keras.layers import Convolution1D, Dense, add, Lambda

import base
from models.layers import tcn_block, graph_propagation


class SpatioTCN(base.NN):
    """
    Spatio Temporal Convolutional Network, mostly similar with TCN, but a graph propagation layer
    is added between each TCN block for message passing across different nodes. The graph
    propagation layers used here is similar to graph convolutional network [1]

    References:
    [1] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks.”

    Remark:
    - The implementation here is highly similar as TCN (apart from an additional graph_propagation
      layer), but we do not refactor and simply use duplicated code here for illustration purpose
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, params, adj_np):
        self.adj_np = adj_np
        super(SpatioTCN, self).__init__(params)

    def _forward(self, inputs):
        x = inputs
        x = Convolution1D(self.params['nb_filters'], 1, padding=self.params['padding'])(x)
        skip_connections = []
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

        # Use graph_propagation layer for message passing across nodes
        x = graph_propagation(x, self.adj_np)

        if self.params['use_skip_connections']:
            x = add(skip_connections)

        if not self.params['return_sequences']:
            x = Lambda(lambda tt: tt[:, -1, :])(x)

        outputs = Dense(5)(x)
        return outputs
