import pandas as pd
from keras.api.utils import set_random_seed
from keras.api.models import load_model
import json

from utils.plots import plot_loss
from utils.models import create_model, train_model, evaluate_model
from utils.processing import preparate_data

train_model_mode = True  # обучаем модель или загружаем уже обученную
repeats = 3  # кол-во повторений обучения
# Установим seed для воспроизводимости результатов:
seed = 42
seeds = [42, 42, 42]

# Прочитаем данные:
df = pd.read_csv("data/ferritin.csv", sep=",")
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]

if train_model_mode:
    list_statistics = []
    for i in range(repeats):
        set_random_seed(seeds[i])
        class_weight, x_train, y_train, x_test, y_test = preparate_data(
            df, targets, seeds[i]
        )
        model = create_model()
        history_data = train_model(
            model,
            x_train,
            y_train,
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

    with open("./output/results.json", "w") as file:
        json.dump(avg_statistics, file, ensure_ascii=False, indent=4)
else:
    class_weight, x_train, y_train, x_test, y_test = preparate_data(df, targets, seed)
    # Загрузим уже обученную модель:
    model = load_model("saved_models/ferritin-10000-3.keras")
    y_train_predict = model.predict(x_train)
    y_predict = model.predict(x_test)
    print(f"Shape of y train predict: {y_train_predict.shape}")
    print(f"Shape of y predict: {y_predict.shape}")

    evaluate_model(
        y_train, y_test, y_train_predict, y_predict, "roc_curve-ferritin-10000-3", True
    )
