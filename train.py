import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from keras.api.utils import set_random_seed
import json
import random
from utils.plots import *
from utils.models import *
from utils.processing import *

repeats = 3  # кол-во повторений обучения
# Установим seed для воспроизводимости результатов:
seeds = [random.randint(0, 2**32 - 1) for _ in range(repeats)]

# Прочитаем данные:
df = pd.read_csv("./data/ferritin-all-calc.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
n_features = len(df.columns) - len(targets)
print(f"{n_features=}")

hidden_units = 90

list_statistics = []
for i in range(repeats):
    # set_random_seed(seeds[i])
    class_weight, x_train, y_train, x_test, y_test, pos, neg = preparate_data(
        df, n_features, targets, scale=True, encode=False, seed=None
    )
    model = create_model(n_features, hidden_units, class_weight, pos, neg)
    history_data = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        class_weight,
        isSave=True,
        filename="ferritin-all",
    )
    # Визуализируем кривые обучения:
    plot_loss2(history_data, False, f"history_100-epochs_ferritin-all")

    y_train_predict = model.predict(x_train)
    y_predict = model.predict(x_test)
    print(f"Shape of y train predict: {y_train_predict.shape}")
    print(f"Shape of y predict: {y_predict.shape}")

    statistics = evaluate_model(
        y_train,
        y_test,
        y_train_predict,
        y_predict,
        "roc_curve-ferritin-all",
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
    f"./output/base_model/res-calc_data-{hidden_units}-units.json",
    "w",
) as file:
    json.dump(avg_statistics, file, ensure_ascii=False, indent=4)
