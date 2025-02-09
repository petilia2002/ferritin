import tensorflow as tf
import keras.backend as K
from keras.losses import Loss


class CustomBinaryCrossentropy(Loss):
    def __init__(self, class_weight):
        super().__init__()
        self.class_weight = class_weight

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weights = y_true * self.class_weight[1] + (1.0 - y_true) * self.class_weight[0]
        if sample_weight is not None:
            loss = loss * weights
        return tf.reduce_mean(loss)


class CustomCategoricalCrossentropy(Loss):
    def __init__(self, class_weight):
        super().__init__()
        self.class_weight = tf.constant(
            [class_weight[i] for i in class_weight.keys()], dtype=tf.float32
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = y_true * tf.math.log(y_pred)
        loss = -tf.reduce_sum(loss, axis=-1)
        weights = (
            y_true[:, -1] * self.class_weight[1]
            + (1.0 - y_true[:, -1]) * self.class_weight[0]
        )
        if sample_weight is not None:
            loss = loss * weights
        return tf.reduce_mean(loss)


__all__ = [
    "CustomBinaryCrossentropy",
    "CustomCategoricalCrossentropy",
]
