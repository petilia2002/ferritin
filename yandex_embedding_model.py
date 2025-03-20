import numpy as np
import keras
from keras.api.models import Model
from keras.api.layers import (
    Input,
    Embedding,
    Reshape,
    Concatenate,
    Flatten,
    Dense,
    Dropout,
)


def create_embedding_model(n_features: int, analytes_dim: int, d_embedding: int):

    emb_input = Input(name="input_analytes", shape=(n_features,))
    emb_layer = Embedding(
        name="embedding", input_dim=analytes_dim, output_dim=d_embedding
    )(emb_input)

    val_input = Input(name="input_values", shape=(n_features,))
    val_layer = Reshape((-1, 1))(val_input)

    merged = Concatenate(axis=-1)([emb_layer, val_layer])
    flat_layer = Flatten()(merged)
    # x = Dense(name="dense_1", units=512, activation="relu")(flat_layer)
    # x = Dropout(name="dropout_1", rate=0.3)(x)
    # x = Dense(name="dense_2", units=256, activation="relu")(x)
    # x = Dropout(name="dropout_2", rate=0.2)(x)
    x = Dense(name="dense_3", units=90, activation="relu")(flat_layer)
    x = Dropout(name="dropout_3", rate=0.1)(x)
    output = Dense(name="dense_4", units=2, activation="softmax")(x)

    model = Model(inputs=[emb_input, val_input], outputs=output)
    opt = keras.optimizers.Adam(learning_rate=0.003)
    l = keras.losses.CategoricalCrossentropy()
    m = keras.metrics.CategoricalAccuracy()

    model.compile(optimizer=opt, loss=l, metrics=[m])
    model.summary()
    return model


n_features = 12
analytes_dim = 38
d_embedding = 256

model = create_embedding_model(n_features, analytes_dim, d_embedding)
