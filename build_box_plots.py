import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

for g in range(len(labels) // 4):
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    for i in range(rows):
        for j in range(cols):
            parameter = labels[g * (rows + cols) + i * rows + j]
            axes[i, j].set_title(f"Box-plot для каждой лаборатории", pad=10)
            axes[i, j].set_xlabel(f"Значение {parameter}", labelpad=1)

            data_to_plot = []
            positions = []
            labs = []
            for k in range(labs_id.size):
                lab_id = labs_id[k]
                lab = f"lab_id{lab_id}"
                data = df[df["lab_id"] == lab_id].filter(items=[parameter])
                data.dropna(inplace=True, ignore_index=True)

                std_value = float((data[parameter].std()))

                if math.isnan(std_value):
                    print(
                        f"Дисперсия NaN у показателя {parameter} в лаборатории {lab_id}"
                    )
                    continue
                data_to_plot.append(data[parameter])
                positions.append(k + 1)
                labs.append(f"Лаб.{lab_id}")

            axes[i, j].boxplot(
                data_to_plot,
                positions=positions,
                vert=False,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                showfliers=False,
            )
            axes[i, j].set_yticks(positions)
            axes[i, j].set_yticklabels(labs)
            axes[i, j].grid(visible=True, linestyle="--", alpha=0.7)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"./browse_data_images/box_plots_page_{g + 1}.png", dpi=300)
    plt.show()
