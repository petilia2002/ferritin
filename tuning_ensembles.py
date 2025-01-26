import pandas as pd
from keras.utils import set_random_seed
import json
import random
from utils.plots import plot_loss
from utils.models import create_model, create_ensemble, train_model, evaluate_model
from utils.processing import preparate_data

# Прочитаем данные:
df = pd.read_csv("./data/ferritin-v2.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]
k = [10, 32, 50, 75]
hidden_units = [10, 30, 50, 75, 90]
list_statistics = []

for _k in k:
    for _units in hidden_units:
        class_weight, x_train, y_train, x_test, y_test = preparate_data(
            df, targets, seed=None
        )
        model = create_ensemble(k=_k, hidden_units=_units)
        history_data = train_model(
            model,
            x_train,
            y_train,
            class_weight,
            isSave=True,
            filename="ferritin-v2-ensemble",
        )
        # Визуализируем кривые обучения:
        plot_loss(history_data, f"history_100-epochs_ensemble")

        y_train_predict = model.predict(x_train)
        y_predict = model.predict(x_test)
        print(f"Shape of y train predict: {y_train_predict.shape}")
        print(f"Shape of y predict: {y_predict.shape}")

        statistics = evaluate_model(
            y_train,
            y_test,
            y_train_predict,
            y_predict,
            "roc_curve-ensemble",
            False,
        )
        s = {}
        s["k"] = _k
        s["hidden_units"] = _units
        s["AUC"] = statistics["auc"]
        list_statistics.append(s)

with open(f"./output/mini_ensembles/tuning_results.json", "w") as file:
    json.dump(list_statistics, file, ensure_ascii=False, indent=4)
