import numpy as np
import tensorflow as tf
from keras.layers import Layer
from keras.initializers import RandomNormal, GlorotUniform
import keras.backend as K


def init_random_signs(shape, dtype=tf.float32):
    bern = K.random_binomial(shape=shape, p=0.5, dtype=dtype)
    return K.cast(bern * 2 - 1, dtype)


def init_random_normal(shape, dtype=tf.float32):
    return tf.random.normal(shape, mean=0.0, stddev=0.5, dtype=dtype)


class DeepEnsembleLayer(Layer):
    def __init__(
        self,
        k: int = 32,
        units: int = 5,
        last_layer: bool = False,
        seed=None,
    ):
        super().__init__()
        self.units = units
        self.k = k
        self.last_layer = last_layer
        self.seed = seed
        self.bias_initializer = RandomNormal(mean=0.0, stddev=0.5, seed=self.seed)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.k, input_shape[-1], self.units),
            initializer=GlorotUniform(seed=self.seed),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.k, 1, self.units),
            initializer=self.bias_initializer,
            trainable=True,
            name="bias",
        )

    def call(self, inputs):  # Inputs must be 3D. For example:
        x = tf.transpose(inputs, perm=[1, 0, 2])
        x = tf.matmul(x, self.kernel)  # (2, 3, 2) x (2, 2, 2) = (2, 3, 2)
        output = tf.add(x, self.bias)  # (2, 3, 2) + (2, 2) = (2, 3, 2)
        output = (
            tf.nn.sigmoid(output) if self.last_layer else tf.nn.relu(output)
        )  # (2, 3, 2)
        if self.last_layer:
            output = tf.transpose(output, perm=[1, 0, 2])  # (3, 2, 2)
            output = tf.reduce_mean(output, axis=1)  # (3, 2)
        else:
            output = tf.transpose(output, perm=[1, 0, 2])  # (3, 2, 2)
        return output


class BatchEnsembleLayer(Layer):
    def __init__(
        self,
        k: int = 32,
        units: int = 5,
        last_layer: bool = False,
        random_signs: bool = False,
        seed=None,
    ):
        super().__init__()
        self.units = units
        self.k = k
        self.last_layer = last_layer
        self.seed = seed
        self.initializer = init_random_signs if random_signs else init_random_normal

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=GlorotUniform(seed=self.seed),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.k, 1, self.units),
            initializer=self.initializer,
            trainable=True,
            name="bias",
        )
        self.r = self.add_weight(
            shape=(self.k, 1, input_shape[-1]),
            initializer=self.initializer,
            trainable=True,
            name="r",
        )
        self.s = self.add_weight(
            shape=(self.k, 1, self.units),
            initializer=self.initializer,
            trainable=True,
            name="s",
        )

    def call(self, inputs):  # Inputs must be 3D. For example:
        x = tf.transpose(inputs, perm=[1, 0, 2])
        x = tf.multiply(x, self.r)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = tf.matmul(x, self.kernel)  # (2, 3, 2) x (2, 2) = (2, 3, 2)
        output = tf.multiply(output, self.s)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = tf.add(output, self.bias)  # (2, 3, 2) + (2, 1, 2) = (2, 3, 2)
        output = (
            tf.nn.sigmoid(output) if self.last_layer else tf.nn.relu(output)
        )  # (2, 3, 2)
        if self.last_layer:
            output = tf.transpose(output, perm=[1, 0, 2])  # (3, 2, 2)
            output = tf.reduce_mean(output, axis=1)  # (3, 2)
        else:
            output = tf.transpose(output, perm=[1, 0, 2])  # (3, 2, 2)
        return output


class MiniEnsembleLayer(Layer):
    def __init__(
        self,
        k: int = 32,
        units: int = 5,
        use_r: bool = True,
        use_s: bool = False,
        last_layer: bool = False,
        random_signs: bool = False,
        seed=None,
    ):
        super().__init__()
        self.k = k
        self.units = units
        self.use_r = use_r
        self.use_s = use_s
        self.last_layer = last_layer
        self.seed = seed
        self.initializer = init_random_signs if random_signs else init_random_normal

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=GlorotUniform(seed=self.seed),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer=self.initializer,
            trainable=True,
            name="bias",
        )
        self.r = (
            self.add_weight(
                shape=(self.k, 1, input_shape[-1]),
                initializer=self.initializer,
                trainable=True,
                name="r",
            )
            if self.use_r
            else None
        )
        self.s = (
            self.add_weight(
                shape=(self.k, 1, self.units),
                initializer=self.initializer,
                trainable=True,
                name="s",
            )
            if self.use_s
            else None
        )

    def call(self, inputs):  # Inputs must be 3D. For example:
        x = tf.transpose(inputs, perm=[1, 0, 2])
        if self.use_r:
            x = tf.multiply(x, self.r)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = tf.matmul(x, self.kernel)  # (2, 3, 2) x (2, 2) = (2, 3, 2)
        if self.use_s:
            output = tf.multiply(output, self.s)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = tf.add(output, self.bias)  # (2, 3, 2) + (2, ) = (2, 3, 2)
        output = (
            tf.nn.sigmoid(output) if self.last_layer else tf.nn.relu(output)
        )  # (2, 3, 2)
        if self.last_layer:
            output = tf.transpose(output, perm=[1, 0, 2])  # (3, 2, 2)
            output = tf.reduce_mean(output, axis=1)  # (3, 2)
        else:
            output = tf.transpose(output, perm=[1, 0, 2])  # (3, 2, 2)
        return output
