import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from keras.utils import set_random_seed
import json
import random
import numpy as np
import keras
import keras.ops as ops
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from utils.plots import *
from utils.models import *
from utils.processing import *

repeats = 3  # кол-во повторений обучения
# Прочитаем данные:
df = pd.read_csv("./data/ferritin-all.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
n_features = len(df.columns) - len(targets)
print(f"{n_features=}")

teacher_units = 512
hidden_units = 90

list_statistics = []
for i in range(repeats):
    class_weight, x_train, y_train, x_test, y_test, pos, neg = preparate_data(
        df, n_features, targets, scale=True, encode=True, seed=None
    )

    # Учитель:
    input = Input(shape=(n_features,))
    x = Dense(units=512, activation="relu")(input)
    x = Dropout(0.2)(x)
    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(units=128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(units=32, activation="relu")(x)
    output = Dense(units=2, activation="softmax")(x)

    teacher = Model(inputs=input, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=0.003)
    l = keras.losses.CategoricalCrossentropy()
    m = keras.metrics.CategoricalAccuracy()

    teacher.compile(optimizer=opt, loss=l, metrics=[m])
    teacher.summary()

    history = teacher.fit(
        x=x_train,
        y=y_train,
        batch_size=500,
        epochs=100,
        verbose=2,
        validation_split=0.2,
        # validation_data=(x_test, y_test),
        shuffle=True,
        class_weight=class_weight,
        # sample_weight=sample_weight,
        callbacks=[reduce_lr()],
    )

    y_train_predict = teacher.predict(x_train)
    y_predict = teacher.predict(x_test)
    print(f"Shape of y train predict: {y_train_predict.shape}")
    print(f"Shape of y predict: {y_predict.shape}")

    teacher_statistics = evaluate_model(
        y_train[:, 0],
        y_test[:, 0],
        y_train_predict[:, 0],
        y_predict[:, 0],
        "roc_curve-ferritin-all",
        False,
    )

    # Ученик:
    input = Input(shape=(n_features,))
    x = Dense(units=hidden_units, activation="relu")(input)
    output = Dense(units=2, activation="softmax")(x)

    student = Model(inputs=input, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=0.003)
    l = keras.losses.CategoricalCrossentropy()
    m = keras.metrics.CategoricalAccuracy()

    student.compile(optimizer=opt, loss=l, metrics=[m])
    student.summary()

    history = student.fit(
        x=x_train,
        y=y_train_predict,
        batch_size=500,
        epochs=100,
        verbose=2,
        validation_split=0.2,
        # validation_data=(x_test, y_test),
        shuffle=True,
        class_weight=class_weight,
        # sample_weight=sample_weight,
        callbacks=[reduce_lr()],
    )

    # Визуализируем кривые обучения:
    plot_loss2(history.history, True, f"history_100-epochs_ferritin-all")

    y_train_predict = student.predict(x_train)
    y_predict = student.predict(x_test)
    print(f"Shape of y train predict: {y_train_predict.shape}")
    print(f"Shape of y predict: {y_predict.shape}")

    y_train = y_train[:, 0]
    y_test = y_test[:, 0]
    y_train_predict = y_train_predict[:, 0]
    y_predict = y_predict[:, 0]
    print(f"Shape of y train predict: {y_train_predict.shape}")
    print(f"Shape of y predict: {y_predict.shape}")

    student_statistics = evaluate_model(
        y_train,
        y_test,
        y_train_predict,
        y_predict,
        "roc_curve-ferritin-all",
        False,
    )
    student_statistics["Teacher AUC"] = teacher_statistics["auc"]
    list_statistics.append(student_statistics)

avg_statistics = {}
for key in list_statistics[0].keys():
    results = list(map(lambda s: s[key], list_statistics))
    avg = sum(results) / len(results)
    avg = round(avg, 2)
    if results[0].is_integer():
        results = list(map(int, results))
    else:
        results = list(map(lambda x: round(x, 2), results))
    field_name = f"avg_{key}"
    avg_statistics[field_name] = avg
    avg_statistics[f"list_{key}"] = results

with open(
    f"./output/distillation/res-all_data-{hidden_units}-units.json",
    "w",
) as file:
    json.dump(avg_statistics, file, ensure_ascii=False, indent=4)
