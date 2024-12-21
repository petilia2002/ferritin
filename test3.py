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

z_score = 1.96
for g in range(len(labels) // 4):
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    for i in range(rows):
        for j in range(cols):
            parameter = labels[g * (rows + cols) + i * rows + j]
            axes[i, j].set_title(f"Box-plot для {parameter}", pad=10)
            axes[i, j].set_xlabel(f"Значение {parameter}", labelpad=1)

            positions = []
            data_to_plot = []

            for k in range(labs_id.size):
                lab = f"lab_id{labs_id[k]}"
                data = df[df["lab_id"] == labs_id[k]].filter(items=[parameter])
                data.dropna(inplace=True, ignore_index=True)

                if data.empty:
                    continue

                q1, q3 = data[parameter].quantile(0.25), data[parameter].quantile(0.75)
                iqr = q3 - q1
                std_value = float((data[parameter].std()))

                low, high = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                # data = data[(data[parameter] > low) & (data[parameter] < high)]

                if math.isnan(std_value):
                    print(
                        f"Дисперсия NaN у показателя {parameter} в лаборатории {labs_id[k]}"
                    )
                    continue

                positions.append(k + 1)  # Позиция для каждого боксплота
                data_to_plot.append(data[parameter])

            # Рисуем boxplot для всех лабораторий
            if data_to_plot:
                axes[i, j].boxplot(
                    data_to_plot,
                    positions=positions,
                    vert=False,
                    patch_artist=True,
                    boxprops=dict(facecolor="lightblue"),
                    showfliers=False,
                )
                axes[i, j].set_yticks(positions)
                axes[i, j].set_yticklabels(
                    [
                        f"Лаб.{labs_id[k]}"
                        for k in range(labs_id.size)
                        if k < len(data_to_plot)
                    ]
                )

            axes[i, j].grid(True, linestyle="--", alpha=0.7)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(
        f"./browse_data_images/box_plots_page_{g + 1}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
