import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd

from utils.statistic import find_confidence_interval

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

# Алгоритм по обрезанию концов распределения
# Сначала строим само распределение:
x = df["age"].values
x = x[~np.isnan(x) & ~np.isinf(x)]
bw_adjust = 1.0

# Находим доверительный интервал:
x1, x2, max_density, max_density_point = find_confidence_interval(x, bw_adjust)
print(f"x1 = {x1:.4f}")
print(f"x2 = {x2:.4f}")

# Визуализация графика плотности и доверительного интервала:
# Строим график
plt.figure(figsize=(10, 5))
plt.title("График плотности распределения для возраста", pad=10)
plt.xlabel("Кол-во лет", labelpad=10)
plt.ylabel("Значение плотности", labelpad=10)
sns.kdeplot(x, fill=True, label="Плотность", bw_adjust=bw_adjust)
sns.kdeplot(
    x,
    fill=True,
    label=f"Дов.интервал: [{x1:.2f}, {x2:.2f}]",
    bw_adjust=bw_adjust,
    clip=(x1, x2),
    color="orange",
    alpha=0.5,
)
plt.scatter(
    x=max_density_point,
    y=max_density,
    color="red",
    s=20,
    zorder=2,
    label=f"Мода.: {max_density_point:.2f} лет",
)
plt.axvline(max_density_point, color="red", linestyle="--")
plt.grid(True)
plt.legend(loc="upper right")
plt.savefig("./browse_data_images/conf-intervals/conf_interval_age.png", dpi=300)
