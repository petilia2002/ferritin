from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle


def use_oversampling(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos_mask = (y == 1.0).flatten()
    neg_mask = ~pos_mask

    pos_x, neg_x = x[pos_mask], x[neg_mask]
    pos_y, neg_y = y[pos_mask], y[neg_mask]

    indices = np.arange(len(pos_x), dtype=int)
    indices = np.random.choice(indices, len(neg_x))

    resampled_pos_x = pos_x[indices]
    resampled_pos_y = pos_y[indices]

    resampled_x = np.concatenate([resampled_pos_x, neg_x], axis=0)
    resampled_y = np.concatenate([resampled_pos_y, neg_y], axis=0)

    order = np.arange(len(resampled_y), dtype=int)
    np.random.shuffle(order)
    resampled_x = resampled_x[order]
    resampled_y = resampled_y[order]
    return resampled_x, resampled_y


def preparate_data(
    df: pd.DataFrame,
    n_features: int,
    targets: List[str],
    scale: bool = True,
    seed: int = 42,
) -> Tuple[Dict[int, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    y = df.filter(items=targets).to_numpy().flatten().astype("float64")

    y = np.array(list(map(lambda x: (1 if x < 12.0 else 0), y)), dtype="int64")
    print(f"Shape of y: {y.shape}")
    print(f"Доля пациентов с низким ферритином: {100 * sum(y) / len(y)}%")
    pos, neg = sum(y), len(y) - sum(y)
    print(f"total={len(y)}")
    print(f"{pos=}")
    print(f"{neg=}")

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
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        # x_scaled = np.clip(a=x_scaled, a_min=-5.0, a_max=5.0)
        print(f"Shape of scaled x: {x_scaled.shape}")
    data_size = x_scaled.shape[0]
    print(f"Размер всех данных: {data_size}")

    # Разделим данные на обучающую и тестовую выборки:
    train_size = int(data_size * 0.8)
    test_size = data_size - train_size

    x_train = np.zeros((train_size, n_features), dtype=float)
    x_test = np.zeros((test_size, n_features), dtype=float)

    y_train = np.zeros((train_size, 1), dtype=float)
    y_test = np.zeros((test_size, 1), dtype=float)

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

    # x_train, y_train = use_oversampling(x_train, y_train)
    print(f"Shape of x_train: {x_train.shape}")
    print(f"Shape of x_test: {x_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    return class_weight, x_train, y_train, x_test, y_test, pos, neg
