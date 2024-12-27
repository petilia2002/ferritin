import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import math

statistic = {}

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

df = pd.read_csv("./data/lab.csv", sep=",", dtype={"sampleno": str, "hgb": float})
print(df.head())
print(df.shape)

labs_id = np.unique(df["lab_id"].values.ravel())
print(labs_id)
print(labs_id.size)

gender_labels = {0: "Женщины", 1: "Мужчины"}
counts_df = df["gender"].value_counts(normalize=True) * 100
counts_df.index = counts_df.index.map(gender_labels)
color_labels = [
    "lightblue" if label == "Мужчины" else "pink" for label in counts_df.index
]

plt.figure(figsize=(12, 6))
counts_df.plot(kind="bar", color=color_labels)
plt.title("Гистограмма распределения пациентов по полу", fontsize=12)
plt.xlabel(f"Пол пациентов", fontsize=10)
plt.xticks(rotation=0)
plt.ylabel(f"Доля пациентов, %", fontsize=10)

legend_men = mpatches.Patch(color="lightblue", label="Мужчины")
legend_women = mpatches.Patch(color="pink", label="Женщины")
plt.legend(handles=[legend_women, legend_men], loc="upper right")

plt.savefig("./browse_data_images/gender_hist.png", dpi=300)

z_score = 1.96
for g in range(len(labels) // 4):
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    for i in range(rows):
        for j in range(cols):
            parameter = labels[g * rows * cols + i * cols + j]
            axes[i, j].set_title(
                f"Распределение показателя {parameter} по лабораториям", pad=10
            )
            axes[i, j].set_xlabel(f"{parameter}", labelpad=1)
            axes[i, j].set_ylabel("Значение плотности", labelpad=10)
            statistic[parameter] = {}

            for k in range(labs_id.size):
                lab = f"lab_id{labs_id[k]}"
                statistic[parameter][lab] = {}
                data = df[df["lab_id"] == labs_id[k]].filter(items=[parameter])
                data.dropna(inplace=True, ignore_index=True)

                min_value = float((data[parameter].min()))
                max_value = float((data[parameter].max()))
                mean_value = float((data[parameter].mean()))
                std_value = float((data[parameter].std()))

                statistic[parameter][lab]["min"] = round(min_value, 4)
                statistic[parameter][lab]["max"] = round(max_value, 4)
                statistic[parameter][lab]["mean"] = round(mean_value, 4)
                statistic[parameter][lab]["std"] = round(std_value, 4)

                low, high = (
                    mean_value - z_score * std_value,
                    mean_value + z_score * std_value,
                )
                data = data[(data[parameter] >= low) & (data[parameter] <= high)]

                if math.isnan(std_value):
                    statistic[parameter][lab]["std"] = round(-1.0, 4)
                    print(
                        f"Дисперсия NaN у показателя {parameter} в лаборатории {labs_id[k]}"
                    )
                    continue

                sns.kdeplot(
                    data[parameter],
                    ax=axes[i, j],
                    fill=True,
                    label=f"Лаб. {labs_id[k]}",
                    bw_adjust=1.0,
                )
            axes[i, j].legend(loc="upper right", fontsize="small", framealpha=0.5)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(
        f"./browse_data_images/lab_dist_page_{g + 1}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

with open("./data/labs.json", "w", encoding="utf-8") as file:
    json.dump(statistic, file, ensure_ascii=False, indent=4)
