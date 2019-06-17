import random
import numpy as np
import tensorflow as tf


def is_cuda_available():
    return tf.test.is_gpu_available(cuda_only=True)


def set_seed(seed):
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def tensor_description(var):
    """
    Returns a compact and informative string about a tensor.
    Args:
      var: A tensor variable.
    Returns:
      a string with type and size, e.g.: (float32 1x8x8x1024).
    
    Referred from:
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/model_analyzer.py
    """
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'
    return description
