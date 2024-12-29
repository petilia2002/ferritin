import pandas as pd
from keras.utils import set_random_seed
from keras.models import load_model

from utils.models import evaluate_model
from utils.processing import preparate_data

# Установим seed для воспроизводимости результатов:
seed = 42
set_random_seed(seed)
targets = ["ferritin"]

# Прочитаем данные:
df = pd.read_csv("data/ferritin.csv", sep=",")
print(df.head())
print(df.shape)
print(df.dtypes)

class_weight, x_train, y_train, x_test, y_test = preparate_data(df, targets, seed)
# Загрузим уже обученную модель:
model = load_model("saved_models/ferritin-10000-3.keras")
y_train_predict = model.predict(x_train)
y_predict = model.predict(x_test)
print(f"Shape of y train predict: {y_train_predict.shape}")
print(f"Shape of y predict: {y_predict.shape}")

evaluate_model(
    y_train, y_test, y_train_predict, y_predict, "roc_curve-ferritin-10000-4", True
)
