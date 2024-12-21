import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

labels = [
    "gender",
    "age",
    "hgb",
    "rbc",
    "mcv",
    "plt",
    "wbc",
    "neut",
    "lymph",
    "eo",
    "baso",
    "mono",
    "ferritin",
]

df = pd.read_csv("data/lab.csv", sep=",", dtype={"sampleno": str, "hgb": float})
print(df.head())
print(df.shape)

age_cleaned = df["age"]
# Вычисление статистик только для 'age'
min_value = age_cleaned.min()
max_value = age_cleaned.max()
mean_value = age_cleaned.mean()
std_dev = age_cleaned.std()
mode_value = age_cleaned.mode()[0]  # Вычисление моды

# Вывод результатов
print(f"Минимум: {min_value}")
print(f"Максимум: {max_value}")
print(f"Среднее: {mean_value}")
print(f"Стандартное отклонение: {std_dev}")

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
axes[0].set_title("Плотность распределения возрастов")
axes[0].set_xlabel("Кол-во лет")
axes[0].set_ylabel("Значение плотности")
sns.kdeplot(
    df["age"],
    ax=axes[0],
    fill=True,
    label="Возраст",
    # bw_adjust=0.5,
    # clip=(min_value, max_value),
)

# Подпись статистик в уголке
text_str = f"$\\mu = {mean_value:.2f}$\n$\\sigma = {std_dev:.2f}$"
axes[0].text(
    0.93,
    0.1,
    text_str,
    transform=axes[0].transAxes,
    fontsize=12,
    verticalalignment="center",
    horizontalalignment="center",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)

axes[0].legend(loc="upper right")
plt.show()


plt.figure(figsize=(12, 6))
# Построение box plot
plt.boxplot(
    df["age"],
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    tick_labels=["Age"],
)
plt.title("Box Plot для возрастов")
plt.xlabel("Кол-во лет")
plt.savefig("./browse_data_images/age_box_plot.png", dpi=300)
plt.show()
