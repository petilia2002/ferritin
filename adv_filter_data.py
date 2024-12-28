import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from utils.statistic import find_confidence_interval

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

lab = pd.read_csv("./data/lab.csv", sep=",", dtype={"sampleno": str, "hgb": float})
print(lab.head())
print(lab.shape)

ferritin = (
    lab[(lab["ferritin"].notna()) & (lab["ferritin"] > 0)]
    .reset_index(drop=True)
    .filter(items=labels)
)
print(ferritin.head())
print(ferritin.shape)
ferritin.dropna(inplace=True, ignore_index=True)
print(ferritin.shape)
print(ferritin.head())
print(ferritin.shape)

labels = labels[1:]
bw_adjust = 1.0
conf_intervals = {}
for label in labels:
    conf_intervals[label] = {}
    x = ferritin[label].to_numpy()
    x1, x2, max_density, max_density_point = find_confidence_interval(x, bw_adjust)
    conf_intervals[label]["x1"] = x1
    conf_intervals[label]["x2"] = x2
    conf_intervals[label]["max_density"] = max_density
    conf_intervals[label]["max_density_point"] = max_density_point

with open("./output/CI.json", "w", encoding="utf-8") as file:
    json.dump(conf_intervals, file, ensure_ascii=False, indent=4)

rows, cols = 2, 2
pages = len(labels) // (rows * cols)  # = 3

for page in range(pages):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    for i in range(rows):
        for j in range(cols):
            ind = page * rows * cols + i * cols + j
            parameter = labels[ind]
            data = ferritin.filter(items=[parameter])
            x = data.to_numpy().ravel()
            x1 = conf_intervals[parameter]["x1"]
            x2 = conf_intervals[parameter]["x2"]
            max_density = conf_intervals[parameter]["max_density"]
            max_density_point = conf_intervals[parameter]["max_density_point"]
            sns.kdeplot(
                x,
                fill=True,
                label="Плотность",
                bw_adjust=bw_adjust,
                ax=axes[i, j],
                color="blue",
            )
            sns.kdeplot(
                x,
                fill=True,
                label=f"Дов.интервал: [{x1:.2f}, {x2:.2f}]",
                bw_adjust=bw_adjust,
                clip=(x1, x2),
                color="orange",
                alpha=0.5,
                ax=axes[i, j],
            )
            axes[i, j].scatter(
                x=max_density_point,
                y=max_density,
                color="red",
                s=10,
                zorder=2,
                label=f"Мода: {max_density_point:.2f}",
            )
            axes[i, j].axvline(max_density_point, color="red", linestyle="--")
            axes[i, j].set_title(f"Плотность распределения показателя {parameter}")
            axes[i, j].set_xlabel(f"Значение {parameter}")
            axes[i, j].set_ylabel(f"Значение плотности")
            axes[i, j].grid(True)
            axes[i, j].legend(loc="upper right", fontsize="small", framealpha=0.5)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"./browse_data_images/conf-intervals/CI_page_{page + 1}.png", dpi=300)

for label in labels:
    low, high = conf_intervals[label]["x1"], conf_intervals[label]["x2"]
    ferritin = ferritin[(ferritin[label] >= low) & (ferritin[label] <= high)]

ferritin.reset_index(drop=True, inplace=True)
print(ferritin.head())
print(ferritin.shape)
ferritin.to_csv("./data/ferritin-v2.csv", sep=",", index=False)

columns_to_check = ferritin.drop(columns=["gender"]).columns

filtered_df = ferritin[(ferritin[columns_to_check] <= 0).any(axis=1)]
print(filtered_df.head())
print(filtered_df.shape)
