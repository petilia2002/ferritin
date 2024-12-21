import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/lab.csv", sep=",", dtype={"sampleno": str, "hgb": float})

labs_id = np.unique(df["lab_id"].values.ravel())
print(labs_id)
print(labs_id.size)

parameter = "wbc"

fig, axes = plt.subplots(1, 1, figsize=(12, 6))
axes.set_title(f"Распределение показателя {parameter} по лабораториям", pad=10)
axes.set_xlabel(f"{parameter}", labelpad=1)
axes.set_ylabel("Значение плотности", labelpad=10)

z_score = 1.96
mean = df[parameter].mean()
std = df[parameter].std()
low, high = mean - z_score * std, mean + z_score * std

for i in range(labs_id.size):
    lab = f"lab_id{labs_id[i]}"
    data = df[
        (df["lab_id"] == labs_id[i]) & (df[parameter] >= low) & (df[parameter] <= high)
    ].filter(items=[parameter])
    sns.kdeplot(
        data[parameter],
        ax=axes,
        fill=True,
        label=f"Лаб. {labs_id[i]}",
    )
axes.legend(loc="upper right")
plt.subplots_adjust(hspace=0.5)
plt.savefig(f"./browse_data_images/wbc_dist_by_labs.png", dpi=300, bbox_inches="tight")
plt.show()
