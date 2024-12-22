import pandas as pd
from keras.api.utils import set_random_seed
from keras.api.models import load_model

from utils.plots import plot_loss
from utils.models import create_model, train_model, evaluate_model
from utils.processing import preparate_data

train_model_mode = True  # обучаем модель или загружаем уже обученную
# Установим seed для воспроизводимости результатов:
seed = 42
set_random_seed(seed)

# Прочитаем данные:
df = pd.read_csv("data/ferritin.csv", sep=",")
print(df.head())
print(df.shape)
print(df.dtypes)

targets = ["ferritin"]

class_weight, x_train, y_train, x_test, y_test = preparate_data(df, targets, seed)

model = create_model()

if train_model_mode:
    # Обучим модель:
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
else:
    # Загрузим уже обученную модель:
    model = load_model("saved_models/ferritin-10000-3.keras")

y_train_predict = model.predict(x_train)
y_predict = model.predict(x_test)
print(f"Shape of y train predict: {y_train_predict.shape}")
print(f"Shape of y predict: {y_predict.shape}")

evaluate_model(
    y_train, y_test, y_train_predict, y_predict, "roc_curve-ferritin-10000-3"
)
