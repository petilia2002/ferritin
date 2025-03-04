# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import Pool, CatBoostClassifier
from lightgbm import LGBMClassifier

sklearn.set_config(enable_metadata_routing=True)

import numpy as np

np.set_printoptions(suppress=True)

import pandas as pd
from keras.api.utils import set_random_seed
import json
import random
from utils.plots import *
from utils.models import *
from utils.processing import *

# Прочитаем данные:
df = pd.read_csv("./data/ferritin-all.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
n_features = len(df.columns) - len(targets)
print(f"{n_features=}")

y = df.filter(items=targets).to_numpy().flatten().astype("float64")
y = np.array(list(map(lambda x: (1 if x < 12.0 else 0), y)), dtype="int64")
class_weight = dict(
    zip(
        np.unique(y),
        compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y),
    )
)

repeats = 10  # кол-во повторов обучения
# model_names = [
#     "logistic_regression",
#     "naive_bayes",
#     "k_neighbors",
#     "decision_tree",
#     "SGD",
#     "random_forest",
#     "XGBoost",
#     "CatBoost",
# ]
# classifiers = [
#     OneVsRestClassifier(
#         LogisticRegression(class_weight=None).set_fit_request(sample_weight=True)
#     ),
#     GaussianNB(),
#     KNeighborsClassifier(n_neighbors=10),
#     DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced"),
#     SGDClassifier(loss="log_loss"),
#     RandomForestClassifier(n_estimators=100),
#     XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5),
#     CatBoostClassifier(
#         iterations=None,
#         learning_rate=0.1,
#         depth=5,
#         loss_function="Logloss",
#         class_weights=class_weight,
#         verbose=True,
#     ),
# ]

model_names = ["LightGBM"]
classifiers = [
    LGBMClassifier(class_weight=class_weight, n_estimators=100, learning_rate=0.1)
]
summary_statistics = []

for j in range(len(classifiers)):
    list_statistics = []
    for i in range(repeats):
        _, x_train, y_train, x_test, y_test, pos, neg = preparate_data(
            df, n_features, targets, scale=True, encode=False, seed=None
        )
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        sample_weight = np.array([class_weight[label] for label in y_train])

        clf = classifiers[j]
        if model_names[j] == "k_neighbors" or model_names[j] == "decision_tree":
            clf.fit(x_train, y_train)
        elif model_names[j] == "CatBoost" or model_names[j] == "LightGBM":
            data = Pool(data=x_train, label=y_train)
            clf.fit(x_train, y_train)
        else:
            clf.fit(x_train, y_train, sample_weight=sample_weight)

        y_train_predict = clf.predict_proba(x_train)
        y_predict = clf.predict_proba(x_test)
        print(f"Shape of y train predict: {y_train_predict.shape}")
        print(f"Shape of y predict: {y_predict.shape}")
        print(y_predict)

        y_train_predict = y_train_predict[:, 1]
        y_predict = y_predict[:, 1]
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

    avg_statistics = {"model_name": model_names[j]}
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
    summary_statistics.append(avg_statistics)

with open(
    f"./output/ml/res-all_data.json",
    "w",
) as file:
    json.dump(summary_statistics, file, ensure_ascii=False, indent=4)
