import tensorflow as tf
from keras.backend import epsilon


def categorical_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, epsilon(), 1.0 - epsilon())
    loss = y_true * tf.math.log(y_pred)
    loss = -tf.reduce_sum(loss, axis=-1)
    return tf.reduce_mean(loss)
