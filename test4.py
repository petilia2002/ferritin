import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np

labels = [
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

df = pd.read_csv("./data/lab.csv", sep=",")
print(df.head())
print(df.shape)

x = df["age"].values
print(x)

# Строим график
plt.figure(figsize=(12, 6))
plt.title("График функции плотности")
plt.xlabel("X")
plt.ylabel("Y")
ax = sns.kdeplot(x, fill=True, label="Плотность", bw_adjust=1.0)

# Используем gaussian_kde для оценки плотности
kde = gaussian_kde(x, bw_method=1.0)

# Интегрируем функцию плотности
xmin, xmax = np.min(x), np.max(x)
print(f"X-min = {xmin}")
print(f"X-max = {xmax}")
integral = kde.integrate_box_1d(xmin, xmax)
print(f"Интеграл (площадь под графиком): {integral}")
total_integral = kde.integrate_box_1d(-np.inf, np.inf)
print(f"Полный интеграл: {total_integral}")

a, b = xmin, xmax

ax = sns.kdeplot(
    x,
    fill=True,
    label=f"Интеграл [{a}, {b}]",
    bw_adjust=1.0,
    clip=(a, b),
    color="orange",
    alpha=0.5,
)

plt.grid(True)
plt.legend(loc="upper right")
plt.show()
