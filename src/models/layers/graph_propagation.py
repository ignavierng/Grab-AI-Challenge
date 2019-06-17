import tensorflow as tf
import tensorflow.keras.backend as K


def graph_propagation(x, adj_np):
    """
    A layer for message passing between different nodes, which simply multiply the tensor by 
    the adjacency matrix [1].

    References:
    [1] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks.”

    Remark:
    - The data passed to x must have first dimention (i.e., batch_size) equivalent to number of nodes,
      which is adj_np.shape[0] or adj_np.shape[1]
    """
    assert adj_np.shape[0] == adj_np.shape[1], 'adj_np must be a square numpy matrix'

    # Convert numpy array to tensor
    adj_tensor = K.variable(adj_np)

    if len(K.int_shape(x)) == 3:
        # Sliding features (LSTM or TCN)
        return tf.einsum('ij, jkl->ikl', adj_tensor, x)
    elif len(K.int_shape(x)) == 2:
        # Not sliding features, MLP
        return K.dot(adj_tensor, x)
    else:
        raise NotImplementedError('Not implemented for current keras tensor shape')
