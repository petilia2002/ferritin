import pandas as pd
import json
from keras.utils import set_random_seed
import random
from utils.plots import plot_loss
from utils.models import create_model, train_model, evaluate_model
from utils.processing import preparate_data

targets = ["ferritin"]  # целевой показатель
repeats = 3  # кол-во повторений обучения

# Прочитаем данные:
df = pd.read_csv("./data/ferritin-all.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

n_features = len(df.columns) - len(targets)
print(f"{n_features=}")

# list_of_units = [u for u in range(5, 105, 5)]
list_of_units = [2**p for p in range(5, 12)]
metrics_by_units = []

for u in list_of_units:
    list_statistics = []
    # seeds = [random.randint(0, 2**32 - 1) for _ in range(repeats)]
    for i in range(repeats):
        # set_random_seed(seeds[i])
        class_weight, x_train, y_train, x_test, y_test = preparate_data(
            df, n_features, targets, scale=True, seed=None
        )
        model = create_model(hidden_units=u)
        history_data = train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            class_weight,
            isSave=True,
            filename="ferritin-10000-3",
        )
        # Визуализируем кривые обучения:
        plot_loss(history_data, f"history_100-epochs_ferritin-10000-3")

        y_train_predict = model.predict(x_train)
        y_predict = model.predict(x_test)
        print(f"Shape of y train predict: {y_train_predict.shape}")
        print(f"Shape of y predict: {y_predict.shape}")

        statistics = evaluate_model(
            y_train,
            y_test,
            y_train_predict,
            y_predict,
            "roc_curve-ferritin-10000-3",
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

    metrics = {}
    metrics["units"] = u

    keys = [key for key in avg_statistics.keys() if key.startswith("avg_")]

    for k in keys:
        metrics[k] = avg_statistics[k]

    metrics_by_units.append(metrics)

with open("./output/base_model/all_data-powers-units.json", "w") as file:
    json.dump(metrics_by_units, file, ensure_ascii=False, indent=4)
