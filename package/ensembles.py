import numpy as np
import tensorflow as tf
from keras.layers import Layer
import keras.ops as ops
from keras.initializers import RandomNormal, GlorotUniform
from keras.random import binomial


def init_random_signs(shape, dtype=None):
    bern = binomial(shape=shape, counts=1, probabilities=0.5, dtype=dtype)
    return ops.add(ops.multiply(bern, 2.0), -1.0)


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

    def build(self, input_shape):
        self.kernel = self.add_weight(
            (self.k, input_shape[-1], self.units),
            initializer=GlorotUniform(seed=self.seed),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            (self.k, 1, self.units),
            initializer=RandomNormal(mean=0.0, stddev=0.5, seed=self.seed),
            trainable=True,
            name="bias",
        )

    def call(self, inputs):  # Inputs must be 3D. For example:
        x = ops.transpose(inputs, (1, 0, 2))
        x = ops.matmul(x, self.kernel)  # (2, 3, 2) x (2, 2, 2) = (2, 3, 2)
        output = ops.add(x, self.bias)  # (2, 3, 2) + (2, 2) = (2, 3, 2)
        output = (
            ops.sigmoid(output) if self.last_layer else ops.relu(output)
        )  # (2, 3, 2)
        if self.last_layer:
            output = ops.transpose(output, (1, 0, 2))  # (3, 2, 2)
            output = ops.average(output, axis=1)  # (3, 2)
        else:
            output = ops.transpose(output, (1, 0, 2))  # (3, 2, 2)
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
        self.initializer = (
            init_random_signs
            if random_signs
            else RandomNormal(mean=0.0, stddev=0.5, seed=self.seed)
        )

    def build(self, input_shape):
        self.kernel = self.add_weight(
            (input_shape[-1], self.units),
            initializer=GlorotUniform(seed=self.seed),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            (self.k, 1, self.units),
            initializer=RandomNormal(mean=0.0, stddev=0.5, seed=self.seed),
            trainable=True,
            name="bias",
        )
        self.r = self.add_weight(
            (self.k, 1, input_shape[-1]),
            initializer=self.initializer,
            trainable=True,
            name="r",
        )
        self.s = self.add_weight(
            (self.k, 1, self.units),
            initializer=self.initializer,
            trainable=True,
            name="s",
        )

    def call(self, inputs):  # Inputs must be 3D. For example:
        x = ops.transpose(inputs, (1, 0, 2))
        x = ops.multiply(x, self.r)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = ops.matmul(x, self.kernel)  # (2, 3, 2) x (2, 2) = (2, 3, 2)
        output = ops.multiply(output, self.s)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = ops.add(output, self.bias)  # (2, 3, 2) + (2, 1, 2) = (2, 3, 2)
        output = output if self.last_layer else ops.relu(output)  # (2, 3, 2)
        if self.last_layer:
            output = ops.transpose(output, (1, 0, 2))  # (3, 2, 2)
            output = ops.average(output, axis=1)  # (3, 2)
        else:
            output = ops.transpose(output, (1, 0, 2))  # (3, 2, 2)
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
        self.initializer = (
            init_random_signs
            if random_signs
            else RandomNormal(mean=0.0, stddev=0.5, seed=self.seed)
        )

    def build(self, input_shape):
        self.kernel = self.add_weight(
            (input_shape[-1], self.units),
            initializer=GlorotUniform(seed=self.seed),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            (self.units,),
            initializer=RandomNormal(mean=0.0, stddev=0.5, seed=self.seed),
            trainable=True,
            name="bias",
        )
        self.r = (
            self.add_weight(
                (self.k, 1, input_shape[-1]),
                initializer=self.initializer,
                trainable=True,
                name="r",
            )
            if self.use_r
            else None
        )
        self.s = (
            self.add_weight(
                (self.k, 1, self.units),
                initializer=self.initializer,
                trainable=True,
                name="s",
            )
            if self.use_s
            else None
        )

    def call(self, inputs):  # Inputs must be 3D. For example:
        x = ops.transpose(inputs, (1, 0, 2))
        if self.use_r:
            x = ops.multiply(x, self.r)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = ops.matmul(x, self.kernel)  # (2, 3, 2) x (2, 2) = (2, 3, 2)
        if self.use_s:
            output = ops.multiply(output, self.s)  # (2, 3, 2) x (2, 1, 2) = (2, 3, 2)
        output = ops.add(output, self.bias)  # (2, 3, 2) + (2, ) = (2, 3, 2)
        output = (
            ops.sigmoid(output) if self.last_layer else ops.relu(output)
        )  # (2, 3, 2)
        if self.last_layer:
            output = ops.transpose(output, (1, 0, 2))  # (3, 2, 2)
            output = ops.average(output, axis=1)  # (3, 2)
        else:
            output = ops.transpose(output, (1, 0, 2))  # (3, 2, 2)
        return output
