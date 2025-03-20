# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import json
from utils.plots import *
from utils.models import *
from utils.processing import *
from yandex_embedding_model import *

repeats = 1  # кол-во повторений обучения

# Прочитаем данные:
df = pd.read_csv("./data/ferritin-all.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
n_features = len(df.columns) - len(targets)
print(f"{n_features=}")
analytes_dim = 38
d_embedding = 256
analytes = np.array([34, 35, 19, 20, 21, 23, 22, 24, 25, 26, 27, 28], dtype=int)

list_statistics = []
for i in range(repeats):
    class_weight, x_train, y_train, x_test, y_test, pos, neg = preparate_data(
        df, n_features, targets, scale=True, encode=True, seed=None
    )
    analytes_repeated = np.tile(analytes, (x_train.shape[0], 1))
    model = create_embedding_model(
        n_features=n_features, analytes_dim=analytes_dim, d_embedding=d_embedding
    )
    history_data = train_yandex_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        analytes_repeated,
        class_weight,
        isSave=True,
        filename="yandex-model",
    )
    # Визуализируем кривые обучения:
    plot_loss2(history_data, True, f"history_100-epochs_ferritin-all")

    y_train_predict = model.predict([analytes_repeated, x_train])
    y_predict = model.predict([np.tile(analytes, (x_test.shape[0], 1)), x_test])
    print(f"Shape of y train predict: {y_train_predict.shape}")
    print(f"Shape of y predict: {y_predict.shape}")

    statistics = evaluate_model(
        y_train[:, 1],
        y_test[:, 1],
        y_train_predict[:, 1],
        y_predict[:, 1],
        "roc_curve-yandex-model",
        False,
    )
    list_statistics.append(statistics)

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
    f"./output/yandex_model/res-all_data.json",
    "w",
) as file:
    json.dump(avg_statistics, file, ensure_ascii=False, indent=4)
