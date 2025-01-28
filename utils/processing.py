from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle


def preparate_data(
    df: pd.DataFrame, targets: List[str], scale: bool = True, seed: int = 42
) -> Tuple[Dict[int, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    y = df.filter(items=targets).to_numpy().flatten().astype("float64")

    y = np.array(list(map(lambda x: (1 if x < 12.0 else 0), y)), dtype="int64")
    print(f"Shape of y: {y.shape}")
    print(f"Доля пациентов с низким ферритином: {100 * sum(y) / len(y)}%")

    x = df.drop(targets, axis=1).to_numpy()
    print(f"Shape of x: {x.shape}")
    print(f"Dtype of x: {x.dtype}")

    # Вычислим веса классов в условиях их несбаланса:
    class_weight = dict(
        zip(
            np.unique(y),
            compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y),
        )
    )
    print(f"Веса классов: {class_weight}")

    x_scaled = x
    # Масштабируем данные:
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_scaled = scaler.fit_transform(x)
        print(f"Shape of scaled x: {x_scaled.shape}")
    data_size = x_scaled.shape[0]
    print(f"Размер всех данных: {data_size}")

    # Разделим данные на обучающую и тестовую выборки:
    train_size = int(data_size * 0.8)
    test_size = data_size - train_size

    x_train = np.zeros((train_size, 12), dtype=float)
    x_test = np.zeros((test_size, 12), dtype=float)

    y_train = np.zeros((train_size, 1), dtype=float)
    y_test = np.zeros((test_size, 1), dtype=float)

    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of x_test: {x_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # Перемешаем данные:
    inds = [i for i in range(data_size)]
    inds = shuffle(inds, random_state=seed)

    for i in range(train_size):
        x_train[i] = x_scaled[inds[i]]
        y_train[i] = y[inds[i]]

    j = 0
    for i in range(train_size, data_size):
        x_test[j] = x_scaled[inds[i]]
        y_test[j] = y[inds[i]]
        j += 1

    return class_weight, x_train, y_train, x_test, y_test
