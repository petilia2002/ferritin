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
            axes[i, j].set_title(f"Box-plot для каждой лаборатории", pad=10)
            axes[i, j].set_xlabel(f"Значение {parameter}", labelpad=1)

            for k in range(labs_id.size):
                lab = f"lab_id{labs_id[k]}"
                data = df[df["lab_id"] == labs_id[k]].filter(items=[parameter])
                data.dropna(inplace=True, ignore_index=True)

                min_value = float((data[parameter].min()))
                max_value = float((data[parameter].max()))
                mean_value = float((data[parameter].mean()))
                std_value = float((data[parameter].std()))

                low, high = (
                    mean_value - z_score * std_value,
                    mean_value + z_score * std_value,
                )
                # data = data[(data[parameter] >= low) & (data[parameter] <= high)]

                if math.isnan(std_value):
                    print(
                        f"Дисперсия NaN у показателя {parameter} в лаборатории {labs_id[k]}"
                    )
                    continue

                axes[i, j].boxplot(
                    data[parameter],
                    vert=False,
                    patch_artist=True,
                    boxprops=dict(facecolor="lightblue"),
                    tick_labels=[f"Лаб.{labs_id[k]}"],
                )
            # axes[i, j].legend(loc="upper right", fontsize="small", framealpha=0.5)
    plt.subplots_adjust(hspace=0.5)
    """
    plt.savefig(
        f"./browse_data_images/lab_dist_page_{g + 1}.png", dpi=300, bbox_inches="tight"
    )
    """
    plt.show()
