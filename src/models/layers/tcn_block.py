from tensorflow.keras.layers import Conv1D, SpatialDropout1D, Activation, BatchNormalization, add


def tcn_block(x, dilation_rate, nb_filters, kernel_size, padding, activation, keep_prob, use_bn):
    # type: (Layer, int, int, int, str, str, float) -> Tuple[Layer, Layer]
    """
    Defines the residual block for the WaveNet TCN

    Args:
        x: The previous layer in the model
        dilation_rate: The dilation power of 2 we are using for this residual block
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        padding: The padding used in the convolutional layers, 'same' or 'causal'.
        activation: The final activation used in o = Activation(x + F(x))
        keep_prob: Float between 0 and 1. Fraction of the input units to keep.
        use_bn: Boolean to indicate if batch_norm should be used
    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.

    Referred from:
    - https://github.com/philipperemy/keras-tcn
    """
    prev_x = x
    for k in range(2):
        x = Conv1D(filters=nb_filters,
                   kernel_size=kernel_size,
                   dilation_rate=dilation_rate,
                   padding=padding)(x)
        if use_bn:
            x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SpatialDropout1D(rate=1-keep_prob)(x)

    # 1x1 conv to match the shapes (channel dimension).
    prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
    res_x = add([prev_x, x])
    res_x = Activation(activation)(res_x)
    return res_x, x
