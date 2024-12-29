from typing import Tuple
import numpy as np
from scipy.stats import gaussian_kde


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


def find_confidence_interval(
    x: np.ndarray, bw_adjust: float
) -> Tuple[float, float, float, float]:
    x = x[~np.isnan(x) & ~np.isinf(x)]
    kde = gaussian_kde(x, bw_method="scott")
    kde.set_bandwidth(kde.factor * bw_adjust)

    # Численно интегрируем кривую плотности:
    S = kde.integrate_box_1d(-np.inf, np.inf)  # должно получиться 1.0
    print(f"S = {S:.4f}")

    # Находим максимум по Y:
    a, b = np.min(x), np.max(x)
    print(a, b)
    x_vals = np.linspace(a, b, 500)
    y_vals = kde(x_vals)

    max_density = np.max(y_vals)
    print(f"Max Y = {max_density:.4f}")

    max_indexes = np.where(y_vals == max_density)[0]
    print(max_indexes)

    max_density_point = x_vals[max_indexes]
    print(max_density_point)

    max_density_point = np.sum(max_density_point) / max_density_point.size

    # Начинаем поиск доверительного интервала:
    x1 = x2 = max_density_point
    dx = 0.005 * max_density_point
    P = 0.95
    p, eps = 0, -0.05
    while p / S < P:
        a, b = x1 - dx, x2 + dx
        left_area = kde.integrate_box_1d(a, x1)
        right_area = kde.integrate_box_1d(x2, b)
        if left_area > right_area and a > eps:
            x1 = a
            p += left_area
        elif left_area == right_area and a > eps:
            x1, x2 = a, b
            p = p + left_area + right_area
        else:
            x2 = b
            p += right_area

    return x1, x2, max_density, max_density_point
