import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

z_score = 1.96
rows, cols = 2, 2
pages = len(labels) // (rows * cols)  # = 3

for page in range(pages):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    for i in range(rows):
        for j in range(cols):
            ind = page * rows * cols + i * rows + j
            parameter = labels[ind]
            data = df.filter(items=[parameter])
            mean_value, std_value = data.mean().item(), data.std().item()
            low, high = (
                mean_value - z_score * std_value,
                mean_value + z_score * std_value,
            )
            """
            q1, q3 = data.quantile(0.25).item(), data.quantile(0.75).item()
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            """
            data = data[(data[parameter] >= low) & (data[parameter] <= high)]
            sns.kdeplot(
                data, ax=axes[i, j], fill=True, label="Плотность", bw_adjust=1.0
            )
            axes[i, j].set_title(f"Плотность распределения показателя {parameter}")
            axes[i, j].set_xlabel(f"{parameter}")
            axes[i, j].set_ylabel(f"Значение плотности")
            axes[i, j].legend(loc="upper right", fontsize="small", framealpha=0.5)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"./browse_data_images/overall_dist_page_{page + 1}.png", dpi=300)
