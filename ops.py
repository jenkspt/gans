import tensorflow as tf
import numpy as np
from functools import partial

def pixelwise_norm(layer, epsilon=1e-8):
    """
    Progressive Growing of Gans 4.2
    Args:
        layer (Tensor): with shape [batch_size, width, height, channel]
        epsilon  (int): Not sure ... assuming it prevents zero values
    """
    with tf.name_scope('pixelwise_norm'):
        return layer / tf.sqrt(tf.reduce_mean(layer**2, axis=-1) + epsilon)

def minibatch_std(batch, shape=(4,4)):
    """
    """
    # shape of batch is [batch_size, width, height, channels]
    with tf.name_scope('minibatch_stddev'):
        channels = tf.concat(batch, -1) 
        # shape is now [width, height, batch_size*channels]
        mean_channel = tf.expand_dims(tf.reduce_mean(channels, axis=-1), axis=-1)
        # Standard deviation for each feature in each spatial location
        batch_std = tf.reduce_mean(tf.sqrt((mean_channel - channels)**2))
        # output shape is [batch_size, width, height, 1]
        return tf.fill([batch.get_shape()[0].value, *shape, 1], batch_std)



dense = partial(tf.layers.dense, activation=tf.nn.leaky_relu, use_bias=False)

def he_init_conv(input_filters, output_filters, kernel, strides):
    """
    Args:
    input_filters (int):
    output_filters (int):

    """
    fan_in = input_filters * np.product(kernel_size)
    fan_out = output_filters * np.product(kernel_size) / np.product(strides)
    filter_std = np.sqrt(4./(fan_in + fan_out))
    return tf.random_normal_initializer(stddev=filter_std)


def conv2d(inputs, filters, kernel_size=3, 
        strides=(1,1), padding='same', 
        activation=tf.nn.leaky_relu,
        use_bias=True,
        kernel_initializer=tf.random_normal_initializer(stddev=1),
        **kwargs):

    conv = tf.layers.conv2d(
            inputs, filters, 
            kernel_size, strides, 
            padding, activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            **kwargs)
    return conv


"""
def conv2d_upsample(inputs, filters, method='linear'):
    if method is 'linear':
        init = tf.constant_initializer([[1,
"""
def conv2d_transpose(inputs, filters, kernel_size=3,
        strides=2, padding='same',
        activation=None, use_bias=True,
        kernel_initializer=tf.random_normal_initializer(stddev=1),
        **kwargs):

    conv = tf.layers.conv2d_transpose(
            inputs=inputs, filters=filters,
            kernel_size=kernel_size, strides=strides,
            padding=padding, activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            **kwargs)
    return conv


def global_avg_pool(input, name='global_avg_pool'):
    """
    Averages each channel to a single value

    Args:
        input (Tensor): 4d input tensor
    Returns:
        Tensor: With shape [batch_size, 1, 1, n_channels]
    """
    shape = input.get_shape().as_list()
    pool = tf.nn.pool(
            input, window_shape=shape[1:-1], 
            pooling_type='AVG', padding='VALID', name=name)
    return pool


def random_labels(batch_size, n_classes, multiclass=False, null_class=True):
    """
    Args:
        batch_size (Tensor or int): Number of examples
        n_classes  (Tensor or int): Number of classes
        multiclass          (bool): If the classes are not mutually exclusive
        null_class          (bool): For multiclass labels, whether to include an 
            extra class in the probability distribution to represent the probability
            that no classes are present
    """
    if multiclass:
        # Assume equal probability for each class 
        # (including the probability that no class is present)
        true_prob = 1/(self.n_classes + 1 if null_class else 0)
        probs = tf.random_uniform([batch_size, n_classes], dtype=tf.float32)
        return tf.where(tf.less(probs, true_prob), 1.0, 0.0)

    indices = tf.random_uniform([batch_size], 0, n_classes, dtype=tf.int32)
    return tf.one_hot(indices, n_classes, dtype=tf.float32)

