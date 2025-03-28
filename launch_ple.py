import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
from keras.utils import set_random_seed
import json
import random

from utils.plots import plot_loss
from utils.processing import preparate_data
from package.embeddings import *
from utils.models import *

# Прочитаем данные:
df = pd.read_csv("./data/ferritin-all.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
n_features = len(df.columns) - len(targets)
print(f"{n_features=}")

k = 10
hidden_units = 90
scale = False
random_signs = False
use_r, use_s = True, True
n_bins = 20
d_embedding = 32

class_weight, x_train, y_train, x_test, y_test, pos, neg = preparate_data(
    df, n_features, targets, scale=scale, encode=False, seed=None
)
bins = compute_bins(x_train, n_bins=n_bins)

model = create_ensemble_with_ple(
    n_features, k=k, hidden_units=hidden_units, bins=bins, d_embedding=d_embedding
)

history_data = train_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    class_weight,
    isSave=False,
    filename="ferritin-v2-ple",
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
    "roc_curve-ple",
    False,
)

with open(
    f"./output/ple_all_data/n_bins-{n_bins}_d-{d_embedding}_k-{k}_units-{hidden_units}.json",
    "w",
) as file:
    json.dump(statistics, file, ensure_ascii=False, indent=4)
