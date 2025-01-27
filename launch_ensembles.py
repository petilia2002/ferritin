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
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]

k = 32
hidden_units = 90

# set_random_seed(seed=42)
class_weight, x_train, y_train, x_test, y_test = preparate_data(df, targets, seed=None)
model = create_ensemble(k=k, hidden_units=hidden_units)
history_data = train_model(
    model,
    x_train,
    y_train,
    class_weight,
    isSave=False,
    filename="ferritin-v2-ensembles",
)
# Визуализируем кривые обучения:
plot_loss(history_data, f"history_100-epochs_ensembles")

y_train_predict = model.predict(x_train)
y_predict = model.predict(x_test)
print(f"Shape of y train predict: {y_train_predict.shape}")
print(f"Shape of y predict: {y_predict.shape}")

statistics = evaluate_model(
    y_train,
    y_test,
    y_train_predict,
    y_predict,
    "roc_curve-ensembles",
    False,
)

with open(
    f"./output/ensembles/k-{k}_units-{hidden_units}-200epochs-2.json",
    "w",
) as file:
    json.dump(statistics, file, ensure_ascii=False, indent=4)
