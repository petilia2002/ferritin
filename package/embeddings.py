import numpy as np
import tensorflow as tf
from keras.layers import Layer
import keras.ops as ops


class LinearEmbeddings(Layer):
    def __init__(self, d_embedding, activation=True):
        super().__init__()
        self.d_embedding = d_embedding
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.d_embedding),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(input_shape[-1], self.d_embedding),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    def call(self, inputs):
        output = ops.add(self.bias, ops.multiply(self.kernel, inputs[..., None]))
        if self.activation:
            output = ops.relu(output)
        return output


class NDense(Layer):
    def __init__(
        self, n_features, out_embedding, initializer="glorot_uniform", is_bias=True
    ):
        super().__init__()
        self.n_features = n_features
        self.out_embedding = out_embedding
        self.initializer = initializer
        self.is_bias = is_bias

    def build(self, input_shape):  # (batch_size, n_features, d_embedding)
        self.kernel = self.add_weight(
            shape=(self.n_features, input_shape[-1], self.out_embedding),
            initializer=self.initializer,
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.n_features, self.out_embedding),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    def call(self, inputs):
        x = ops.transpose(inputs, (1, 0, 2))
        output = ops.matmul(x, self.kernel)
        output = ops.transpose(output, (1, 0, 2))
        if self.is_bias:
            output = ops.add(output, self.bias)
        return output


def compute_bins(x: np.ndarray, n_bins: int = 4) -> list[np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, n_bins)
    bins = [np.unique(feat) for feat in np.quantile(x, quantiles, axis=0).T]
    return bins


class PiecewiseLinearEncoding(Layer):
    def __init__(self, bins: list[np.ndarray]):
        super().__init__()

        n_bins = [(arr.size - 1) for arr in bins]
        self.max_n_bins = max(n_bins)

        self.kernel = self.add_weight(
            shape=(len(n_bins), self.max_n_bins),
            initializer="zeros",
            trainable=False,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(len(n_bins), self.max_n_bins),
            initializer="zeros",
            trainable=False,
            name="bias",
        )

        dif_bins = all([n_bin == n_bins[0] for n_bin in n_bins])  # True OR False
        self.dif_mask = (
            None
            if dif_bins
            else np.vstack(
                [
                    np.hstack(
                        [
                            np.ones((n_bin - 1,), dtype=bool),
                            np.zeros((self.max_n_bins - n_bin,), dtype=bool),
                            np.ones((1,), dtype=bool),
                        ]
                    )
                    for n_bin in n_bins
                ]
            )
        )
        self.dif_mask = ops.convert_to_tensor(self.dif_mask)

        weights = np.zeros((len(n_bins), self.max_n_bins), dtype=float)
        bias = np.zeros((len(n_bins), self.max_n_bins), dtype=float)
        for i, bin_edges in enumerate(bins):
            bin_width = np.diff(bin_edges)
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width

            weights[i, : n_bins[i] - 1] = w[:-1]
            bias[i, : n_bins[i] - 1] = b[:-1]

            weights[i, -1:] = w[-1]
            bias[i, -1:] = b[-1]
        self.set_weights([weights, bias])

    def call(self, inputs):
        output = ops.add(self.bias, ops.multiply(self.kernel, inputs[..., None]))
        return ops.clip(x=output, x_min=0.0, x_max=1.0)


class PiecewiseLinearEmbeddings(Layer):
    def __init__(self, bins: list[np.ndarray]):
        super().__init__()
        self.ple_enc = PiecewiseLinearEncoding(bins=bins)

    def call(self, inputs):
        x = self.ple_enc(inputs)
        out = (
            ops.reshape(x, (x.shape[0], -1))
            if self.ple_enc.dif_mask is None
            else tf.boolean_mask(x, self.ple_enc.dif_mask, axis=1)
        )
        return out


class PiecewiseLinearTrainableEmbeddings(Layer):
    def __init__(
        self,
        bins: list[np.ndarray],
        d_embedding: int,
        linear: bool = True,
        linear_activation: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        self.n_features = len(bins)

        self.linear_emb = (
            LinearEmbeddings(d_embedding, linear_activation) if linear else None
        )
        self.ple_enc = PiecewiseLinearEncoding(bins=bins)

        initializer = "zeros" if linear else "glorot_uniform"
        self.n_dense = NDense(
            self.n_features, d_embedding, initializer=initializer, is_bias=not linear
        )
        self.activation = activation

    def call(
        self, inputs
    ):  # X: (batch_size, n_features) -> (batch_size, n_features, d_embedding)
        x_linear = (
            self.linear_emb(inputs) if self.linear_emb else None
        )  # (batch_size, n_features, d_embedding)

        x_ple = self.ple_enc(inputs)  # (batch_size, n_features, max_n_bins)
        x_ple = self.n_dense(x_ple)  # (batch_size, n_features, d_embedding)

        if self.activation:
            x_ple = ops.relu(x_ple)  # (batch_size, n_features, d_embedding)

        return ops.add(x_ple, x_linear) if x_linear is not None else x_ple
