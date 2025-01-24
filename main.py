from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import set_random_seed
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_model_mode = True  # обучаем модель или загружаем уже обученную
# Установим seed для воспроизводимости результатов:
seed = 42
set_random_seed(seed)

# Прочитаем данные:
df = pd.read_csv("data/ferritin-10000.csv", sep=",")

print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
y = df.filter(items=targets).to_numpy().flatten().astype("float64")

y = np.array(list(map(lambda x: (1 if x < 12.0 else 0), y)), dtype="int64")
print(f"Shape of y: {y.shape}")
print(f"Dtype of y: {y.dtype}")
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

# Масштабируем данные:
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

# Создадим простую модель нейронной сети:
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

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
    start_from_epoch=0,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5,
    verbose=1,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-6,
)

if train_model_mode:
    # Обучим модель:
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
    # Сохраним ее в файл:
    model.save("saved_models/ferritin-10000.keras")
    # Визуализируем кривые обучения:
    history_data = history.history
    train_loss = history_data["loss"]
    val_loss = history_data["val_loss"]
    train_accuracy = history_data["binary_accuracy"]
    val_accuracy = history_data["val_binary_accuracy"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))  # 2 строки, 1 столбец
    x_values = [i for i in range(1, len(train_loss) + 1)]
    # Первый график:
    axes[0, 0].plot(x_values, train_loss, label="Ошибка", color="skyblue")
    axes[0, 0].set_title("Функция потерь при обучении модели", pad=10)
    axes[0, 0].set_xlabel("Эпохи", labelpad=0)
    axes[0, 0].set_ylabel("Функция потерь", labelpad=10)
    axes[0, 0].legend()
    # Второй график:
    axes[1, 0].plot(x_values, train_accuracy, label="Точность", color="tomato")
    axes[1, 0].set_title("Точность модели во время обучения", pad=10)
    axes[1, 0].set_xlabel("Эпохи", labelpad=0)
    axes[1, 0].set_ylabel("Значение точности", labelpad=10)
    axes[1, 0].legend()
    # Третий график:
    axes[0, 1].plot(x_values, val_loss, label="Ошибка", color="skyblue")
    axes[0, 1].set_title("Функция потерь на проверочной выборке", pad=10)
    axes[0, 1].set_xlabel("Эпохи", labelpad=0)
    axes[0, 1].set_ylabel("Функция потерь", labelpad=10)
    axes[0, 1].legend()
    # Четвертый график:
    axes[1, 1].plot(x_values, val_accuracy, label="Точность", color="tomato")
    axes[1, 1].set_title("Точность модели на проверочной выборке", pad=10)
    axes[1, 1].set_xlabel("Эпохи", labelpad=1)
    axes[1, 1].set_ylabel("Значение точности", labelpad=10)
    axes[1, 1].legend()

    # Настройка отступов между графиками:
    fig.subplots_adjust(hspace=0.5)  # Отступ между графиками

    # Сохраняем график:
    plt.savefig(
        "train_images/history_100-epochs_ferritin-10000.png",
        dpi=300,
        # bbox_inches="tight",
    )
    # plt.show()
else:
    # Загрузим уже обученную модель:
    model = load_model("saved_models/ferritin-10000.keras")

y_train_predict = model.predict(x_train)
y_predict = model.predict(x_test)
print(f"Shape of y train predict: {y_train_predict.shape}")
print(f"Shape of y predict: {y_predict.shape}")

# Вычисляем статистику сначала на обучающей выборке для определния оптимального порога:
fpr, tpr, thresholds = roc_curve(y_train, y_train_predict)

train_auc = auc(fpr, tpr)
print(f"Train AUC = {train_auc}")


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


# Вычислим финальную статистику модели:
ot = find_optimal_threshold(tpr, fpr, thresholds)
print(f"Optimal threshold: {ot}")

tp, tn, fp, fn = calculate_confusion_matrix(y_test, y_predict, ot)

accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1_score = 2.0 * tp / (2.0 * tp + fp + fn)

print(f"Statistics:")
print(f"TP = {int(tp)}")
print(f"TN = {int(tn)}")
print(f"FP = {int(fp)}")
print(f"FN = {int(fn)}")

print(f"Accuracy = {accuracy}")
print(f"Sensitivity = {sensitivity}")
print(f"Specificity = {specificity}")
print(f"F1-score = {f1_score}")

# Построим ROC-кривую:
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
test_auc = auc(fpr, tpr)
print(f"AUC = {test_auc}")

plt.figure(figsize=(8, 4))
plt.title(f"ROC-кривая", fontsize=12, pad=10)
plt.xlabel(f"1 - специфичность", fontsize=10)
plt.ylabel(f"чувствительность", fontsize=10, labelpad=10)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.plot(fpr, tpr, color="tomato")
plt.plot([0, 1], [0, 1], color="skyblue", linestyle="--")
plt.plot([0, 1], [1, 0], color="skyblue", linestyle="--")
plt.savefig("train_images/roc_curve-ferritin-10000.png", dpi=300, bbox_inches="tight")
# plt.show()
