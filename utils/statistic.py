from typing import Tuple
import numpy as np


def find_optimal_threshold(
    tpr: np.ndarray, fpr: np.ndarray, thresholds: np.ndarray
) -> float:

    ot1 = ot2 = 0.0
    m1, m2 = -10, 10

    for i in range(tpr.size):
        dif = tpr[i] - (1 - fpr[i])  # разница между sensitivity и specificity
        if dif >= 0:
            if dif < m2:
                m2 = dif
                ot2 = thresholds[i]
        else:
            if dif > m1:
                m1 = dif
                ot1 = thresholds[i]

    ot = (ot1 + ot2) / 2
    return ot


def calculate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> Tuple[float, float, float, float]:

    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten()

    tp = tn = fp = fn = 0.0

    for i in range(y_true.size):
        if y_true[i] == 1 and y_pred[i] >= threshold:
            tp += 1
        if y_true[i] == 0 and y_pred[i] < threshold:
            tn += 1
        if y_true[i] == 0 and y_pred[i] >= threshold:
            fp += 1
        if y_true[i] == 1 and y_pred[i] < threshold:
            fn += 1

    return tp, tn, fp, fn
