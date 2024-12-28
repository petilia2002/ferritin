import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

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
plt.title("График функции")
plt.xlabel("X")
plt.ylabel("Y")
ax = sns.kdeplot(x, fill=True, label="Плотность", bw_adjust=1.0)
plt.grid(True)
plt.legend(loc="upper right")
plt.show()
