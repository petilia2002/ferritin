from typing import Dict, List
import os
import sys
import numpy as np
from sklearn.metrics import roc_curve, auc
import keras
from keras.layers import Input, Dense, Dropout, RepeatVector, Flatten, Layer
from keras.models import Model
from keras.utils import plot_model

# Добавляем корневую папку utils в sys.path:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Добавляем корневую папку проекта в sys.path:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.callbacks import early_stopping, reduce_lr
from utils.plots import plot_roc_curve
from utils.statistic import find_optimal_threshold, calculate_confusion_matrix
from package.embeddings import *
from package.ensembles import *


def create_model(hidden_units: int, units: int) -> Model:
    input = Input(shape=(12,))
    x = Dense(units=hidden_units, activation="relu")(input)
    # x = Dropout(0.3)(x)
    # x = Dense(units=units, activation="relu")(x)
    # x = Dropout(0.3)(x)
    output = Dense(units=1, activation="sigmoid")(x)

    model = Model(inputs=input, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=0.003)
    l = keras.losses.BinaryCrossentropy()
    m = keras.metrics.BinaryAccuracy()

    model.compile(optimizer=opt, loss=l, metrics=[m])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../model_images")
    path = os.path.join(output_dir, f"model.png")
    plot_model(
        model=model,
        to_file=path,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=False,
        show_layer_activations=False,
        show_trainable=True,
        rankdir="TB",
    )
    model.summary()

    return model


def create_ensemble(
    k: int = 3,
    hidden_units: int = 5,
    random_signs: bool = False,
    use_r: bool = True,
    use_s: bool = False,
):
    input = Input(shape=(12,))
    x = RepeatVector(k)(input)
    # x = DeepEnsembleLayer(k=k, units=hidden_units, last_layer=False, seed=None)(x)
    # output = DeepEnsembleLayer(k=k, units=1, last_layer=True, seed=None)(x)

    x = BatchEnsembleLayer(
        k=k,
        units=hidden_units,
        last_layer=False,
        random_signs=random_signs,
        seed=None,
    )(x)
    output = BatchEnsembleLayer(
        k=k,
        units=1,
        last_layer=True,
        random_signs=random_signs,
        seed=None,
    )(x)

    # x = MiniEnsembleLayer(
    #     k=k,
    #     units=hidden_units,
    #     use_r=use_r,
    #     use_s=use_s,
    #     last_layer=False,
    #     random_signs=random_signs,
    #     seed=None,
    # )(x)
    # output = MiniEnsembleLayer(
    #     k=k,
    #     units=1,
    #     use_r=use_r,
    #     use_s=use_s,
    #     last_layer=True,
    #     random_signs=random_signs,
    #     seed=None,
    # )(x)

    model = Model(inputs=input, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=0.003)
    l = keras.losses.BinaryCrossentropy()
    m = keras.metrics.BinaryAccuracy()

    model.compile(optimizer=opt, loss=l, metrics=[m])
    model.summary()
    return model


def create_ensemble_with_ple(
    n_features: int = 12,
    k: int = 3,
    hidden_units: int = 5,
    bins: list[np.ndarray] = [],
    d_embedding: int = 4,
) -> Model:
    input = Input(shape=(n_features,))

    x = PiecewiseLinearTrainableEmbeddings(
        bins=bins,
        d_embedding=d_embedding,
        linear=False,
        linear_activation=True,
        activation=True,
    )(input)
    x = Flatten()(x)

    # x = PiecewiseLinearEmbeddings(bins=bins)(input)
    x = RepeatVector(k)(x)
    x = BatchEnsembleLayer(
        k=k,
        units=hidden_units,
        last_layer=False,
        random_signs=False,
        seed=None,
    )(x)
    output = BatchEnsembleLayer(
        k=k,
        units=1,
        last_layer=True,
        random_signs=False,
        seed=None,
    )(x)

    model = Model(inputs=input, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=0.003)
    l = keras.losses.BinaryCrossentropy()
    m = keras.metrics.BinaryAccuracy()

    model.compile(optimizer=opt, loss=l, metrics=[m])
    model.summary()
    return model


def train_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_weight: Dict[int, float],
    isSave: bool,
    filename: str = "",
) -> Dict[str, List[float]]:
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=500,
        epochs=100,
        verbose=2,
        validation_split=0.2,
        # validation_data=(x_test, y_test),
        shuffle=True,
        class_weight=class_weight,
        callbacks=[early_stopping(), reduce_lr()],
    )
    if isSave:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "../saved_models")
        model.save(os.path.join(output_dir, f"{filename}.keras"))

    return history.history


def evaluate_model(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_predict: np.ndarray,
    y_predict: np.ndarray,
    roc_curve_name: str,
    verbose: bool = False,
) -> Dict[str, float]:
    # Вычисляем статистику сначала на обучающей выборке для определния оптимального порога:
    fpr, tpr, thresholds = roc_curve(y_train, y_train_predict)

    train_auc = auc(fpr, tpr)
    print(f"Train AUC = {train_auc}")

    # Вычислим финальную статистику модели:
    ot = find_optimal_threshold(tpr, fpr, thresholds)
    tp, tn, fp, fn = calculate_confusion_matrix(y_test, y_predict, ot)

    accuracy = 100.0 * (tp + tn) / (tp + tn + fp + fn)
    sensitivity = 100.0 * tp / (tp + fn)
    specificity = 100.0 * tn / (tn + fp)
    f1_score = 100.0 * 2.0 * tp / (2.0 * tp + fp + fn)

    # Построим ROC-кривую:
    fpr, tpr, thresholds = roc_curve(y_test, y_predict)
    test_auc = 100.0 * auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, filename=roc_curve_name)

    if verbose:
        print(f"Statistics:")
        print(f"Optimal threshold: {ot:.2f}")
        print(f"TP = {int(tp)}")
        print(f"TN = {int(tn)}")
        print(f"FP = {int(fp)}")
        print(f"FN = {int(fn)}")
        print(f"Accuracy = {accuracy:.2f} %")
        print(f"Sensitivity = {sensitivity:.2f} %")
        print(f"Specificity = {specificity:.2f} %")
        print(f"F1-score = {f1_score:.2f} %")
        print(f"AUC = {test_auc:.2f} %")

    statistics = {}
    statistics["ot"] = ot
    statistics["tp"] = tp
    statistics["tn"] = tn
    statistics["fp"] = fp
    statistics["fn"] = fn
    statistics["acc"] = accuracy
    statistics["sen"] = sensitivity
    statistics["spec"] = specificity
    statistics["f1"] = f1_score
    statistics["auc"] = test_auc
    return statistics
