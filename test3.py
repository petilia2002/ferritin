import numpy as np
from keras import ops
from keras.losses import CategoricalCrossentropy
import tensorflow as tf
from keras.backend import epsilon

loss = CategoricalCrossentropy()

y_true = np.array([[0, 1, 0], [0, 0, 1]])
y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
# y_true = np.array([[0, 1, 0]])
# y_pred = np.array([[0.05, 0.95, 0]])
print(y_true)
print(y_pred)

res = loss(y_true, y_pred)
print(res)


# def categorical_crossentropy(y_true, y_pred):
#     y_pred = np.clip(a=y_pred, a_min=epsilon(), a_max=1.0 - epsilon())
#     loss = y_true * ops.log(y_pred)
#     loss = -ops.sum(loss, axis=-1)
#     return ops.mean(loss)


def categorical_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, epsilon(), 1.0 - epsilon())
    loss = y_true * tf.math.log(y_pred)
    loss = -tf.reduce_sum(loss, axis=-1)
    return tf.reduce_mean(loss)


res = categorical_crossentropy(y_true, y_pred)
print(res)
