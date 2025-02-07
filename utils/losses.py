import tensorflow as tf
import keras.backend as K
from keras.losses import Loss


# class CustomBinaryCrossentropy(Loss):
#     def __init__(self, class_weight):
#         super().__init__()
#         self.class_weight = class_weight

#     def call(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
#         # weights = y_true * self.class_weight[1] + (1.0 - y_true) * self.class_weight[0]
#         loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
#         # return tf.reduce_mean(weights * loss) if sample_weight else tf.reduce_mean(loss)
#         return tf.reduce_mean(loss)


class CustomBinaryCrossentropy(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred, sample_weight=None):
        # Обрезаем предсказания, чтобы избежать log(0)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Вычисляем стандартную BCE
        loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        if sample_weight is not None:
            loss *= sample_weight
        return tf.reduce_mean(loss)


def binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    return tf.reduce_mean(loss)


def make_weighted_bce(class_weight):
    def weighted_bce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        weights = y_true * class_weight[1] + (1.0 - y_true) * class_weight[0]
        loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(weights * loss)

    return weighted_bce


def categorical_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    loss = y_true * tf.math.log(y_pred)
    loss = -tf.reduce_sum(loss, axis=-1)
    return tf.reduce_mean(loss)


__all__ = [
    "CustomBinaryCrossentropy",
    "binary_crossentropy",
    "make_weighted_bce",
    "categorical_crossentropy",
]
