import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/lab.csv", sep=",", dtype={"sampleno": str, "hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)

age_cleaned = df["age"]
# Вычисление статистик только для 'age'
min_value = age_cleaned.min()
max_value = age_cleaned.max()
mean_value = age_cleaned.mean()
std_dev = age_cleaned.std()

# Вывод результатов
print(f"Минимум: {min_value}")
print(f"Максимум: {max_value}")
print(f"Среднее: {mean_value}")
print(f"Стандартное отклонение: {std_dev}")

fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=False)
axes.set_title("Плотность распределения возрастов")
axes.set_xlabel("Кол-во лет")
axes.set_ylabel("Значение плотности")
sns.kdeplot(
    df["age"],
    ax=axes,
    fill=True,
    label="Возраст",
    # bw_adjust=0.5,
    # clip=(min_value, max_value),
)
axes.legend(loc="upper right")
plt.show()


df1 = df[df["age"] == 110]
print(df1.head())
print(df.shape[0] == df1.shape[0])
