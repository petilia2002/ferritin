import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from keras.utils import set_random_seed
import json
import random
from utils.plots import plot_loss
from utils.models import (
    create_model,
    create_ensemble,
    create_ensemble_with_ple,
    train_model,
    evaluate_model,
)
from utils.processing import preparate_data
from package.embeddings import *

# Прочитаем данные:
df = pd.read_csv("./data/ferritin-v2.csv", sep=",", dtype={"hgb": float})
# df.drop(columns=["gender"], inplace=True)
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
n_features = len(df.columns) - len(targets)
print(f"{n_features=}")

# k = 75
# hidden_units = 90
# n_bins = [5, 10, 12, 15, 20]
# d_embedding = [5, 10, 15, 20, 30]

k = 32
hidden_units = 30
n_bins = [10]
d_embedding = [5]

list_statistics = []

for n in n_bins:
    for d in d_embedding:
        class_weight, x_train, y_train, x_test, y_test = preparate_data(
            df, n_features, targets, scale=False, seed=None
        )
        bins = compute_bins(x_train, n_bins=n)
        print(f"{bins=}")
        model = create_ensemble_with_ple(
            n_features=n_features,
            k=k,
            hidden_units=hidden_units,
            bins=bins,
            d_embedding=d,
        )
        history_data = train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            class_weight,
            isSave=False,
            filename="ferritin-v2-train_ple",
        )
        # Визуализируем кривые обучения:
        plot_loss(history_data, f"history_100-epochs_ple")

        y_train_predict = model.predict(x_train)
        y_predict = model.predict(x_test)
        print(f"Shape of y train predict: {y_train_predict.shape}")
        print(f"Shape of y predict: {y_predict.shape}")

        statistics = evaluate_model(
            y_train,
            y_test,
            y_train_predict,
            y_predict,
            "roc_curve-tr-ple",
            False,
        )
        s = {}
        s["n_bins"] = n
        s["d_embedding"] = d
        s["k"] = k
        s["hidden_units"] = hidden_units
        s["AUC"] = statistics["auc"]
        list_statistics.append(s)

with open(f"./output/ple_ensembles/tuning_non-trainable.json", "w") as file:
    json.dump(list_statistics, file, ensure_ascii=False, indent=4)
