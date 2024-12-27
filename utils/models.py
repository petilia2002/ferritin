from typing import Dict, List
import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import keras
from keras.api.layers import Input, Dense
from keras.api.models import Model

from utils.callbacks import early_stopping, reduce_lr
from utils.plots import plot_roc_curve
from utils.statistic import find_optimal_threshold, calculate_confusion_matrix


def create_model() -> Model:
    input = Input(name="input_1", shape=(12,))
    x = Dense(name="dense_1", units=5, activation="relu")(input)
    # x = Dropout(name="dropout_1", rate=0.1)(x)
    output = Dense(name="dense_2", units=1, activation="sigmoid")(x)

    model = Model(inputs=input, outputs=output)

    opt = keras.optimizers.Adam(learning_rate=0.001)
    l = keras.losses.BinaryCrossentropy()
    m = keras.metrics.BinaryAccuracy()

    model.compile(optimizer=opt, loss=l, metrics=[m])
    model.summary()

    return model


def train_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Dict[int, float],
    isSave: bool,
    filename: str = "",
) -> Dict[str, List[float]]:
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=250,
        epochs=100,
        verbose=2,
        validation_split=0.2,
        shuffle=True,
        class_weight=class_weight,
        callbacks=[early_stopping, reduce_lr],
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
