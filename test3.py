import numpy as np
from keras import ops
from keras.losses import BinaryCrossentropy
import tensorflow as tf
from keras.backend import epsilon

loss = BinaryCrossentropy()

y_true = np.array([[0, 0, 1, 1]])
y_pred = np.array([[0.6, 0.24, 0.97, 0.78]])
print(y_true)
print(y_pred)

res = loss(y_true, y_pred)
print(res)


def binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, epsilon(), 1.0 - epsilon())
    loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    return tf.reduce_mean(loss)


res = binary_crossentropy(y_true, y_pred)
print(res)
