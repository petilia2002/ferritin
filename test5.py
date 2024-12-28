import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd

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
print(x.shape)

x = x[~np.isnan(x) & ~np.isinf(x)]
print(x)
print(x.shape)

bw_adjust = 1.0

plt.figure(figsize=(10, 5))
# Создание графика с sns.kdeplot
sns.kdeplot(x, fill=True, label="Плотность (Seaborn)", bw_adjust=bw_adjust)

# Создание объекта kde с учетом bw_adjust
kde = gaussian_kde(x, bw_method="scott")  # Базовый метод
kde.set_bandwidth(kde.factor * bw_adjust)  # Учитываем bw_adjust

padding = 0.05 * (x.max() - x.min())
# Создание массива точек для графика
a, b = x.min() - padding, x.max() + padding
x_vals = np.linspace(a, b, 500)
y_vals = kde(x_vals)

# Построение графика для объекта kde
plt.plot(x_vals, y_vals, label="Плотность (Scipy)", color="red")

# Оформление графика
plt.legend()
plt.xlabel("x")
plt.ylabel("Плотность")
plt.title("Сравнение KDE из Seaborn и Scipy")
plt.show()
