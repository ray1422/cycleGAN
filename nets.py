import tensorflow as tf
import keras.layers as layers


def generator(x, scope):
    with tf.variable_scope(name_or_scope=scope):
        x = layers.Conv2D(kernel_size=5, filters=64, activation=tf.nn.leaky_relu, padding='SAME')(x)
        x = layers.Conv2D(kernel_size=3, filters=128, strides=2, activation=tf.nn.leaky_relu, padding='SAME')(x)
        x = layers.Conv2D(kernel_size=3, filters=256, strides=2, activation=tf.nn.leaky_relu, padding='SAME')(x)

        for i in range(6):
            x = _res_block(x)

        x = tf.image.resize(x, size=tf.shape(x)[1:3] * 2)
        x = layers.Conv2D(kernel_size=3, filters=128, activation=tf.nn.leaky_relu, padding='SAME')(x)
        x = tf.image.resize(x, size=tf.shape(x)[1:3] * 2)
        x = layers.Conv2D(kernel_size=3, filters=64, activation=tf.nn.leaky_relu, padding='SAME')(x)
        x = layers.Conv2D(kernel_size=5, filters=3, activation=None, padding='SAME')(x)

    return x


def discriminator(x, scope):
    with tf.variable_scope(name_or_scope=scope):
        x = layers.Conv2D(kernel_size=4, filters=32, strides=2, activation=tf.nn.leaky_relu)(x)

        x = layers.Conv2D(kernel_size=4, filters=64, strides=2, activation=None)(x)
        x = tf.nn.leaky_relu(instance_norm(x))

        x = layers.Conv2D(kernel_size=4, filters=128, strides=2, activation=None)(x)
        x = tf.nn.leaky_relu(instance_norm(x))

        x = layers.Conv2D(kernel_size=4, filters=256, strides=2, activation=None)(x)
        x = tf.nn.leaky_relu(instance_norm(x))

        x = layers.Conv2D(kernel_size=4, filters=512, strides=2, activation=None)(x)
        x = tf.nn.leaky_relu(instance_norm(x))

        x = layers.Conv2D(kernel_size=4, filters=1, activation=None)(x)

        return x


def _res_block(x, filters=256):
    y = layers.Conv2D(kernel_size=3, filters=filters, activation=tf.nn.leaky_relu, padding='SAME')(x)
    y = layers.Conv2D(kernel_size=3, filters=filters, activation=tf.nn.leaky_relu, padding='SAME')(y) + x

    return y


def instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mu)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, sigma_sq)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** .5

    return scale * normalized + shift
