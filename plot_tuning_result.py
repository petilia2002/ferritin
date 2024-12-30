import json
import matplotlib.pyplot as plt
import seaborn as sns

with open("./output/metrics_by_units.json", "r", encoding="utf-8") as file:
    metrics_by_units = json.load(file)

print(metrics_by_units)

metrics = ["avg_sen", "avg_spec", "avg_f1", "avg_auc"]
metric_names = [
    "чувствительность",
    "специфичность",
    "f1-мера",
    "AUC-метрика",
]

units = [i["units"] for i in metrics_by_units]

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
rows, cols = 2, 2
for i in range(rows):
    for j in range(cols):
        ind = i * cols + j
        metric = metrics[ind]
        metric_name = metric_names[ind]
        y_values = [i[metric] for i in metrics_by_units]
        axes[i, j].set_title(
            "Зависимость между числом нейронов и качеством модели", pad=10, fontsize=10
        )
        axes[i, j].set_xlabel(f"Кол-во нейронов", labelpad=10, fontsize=8)
        axes[i, j].set_ylabel(f"{metric_name}", labelpad=10, fontsize=8)
        sns.lineplot(
            x=units,
            y=y_values,
            ax=axes[i, j],
            marker="o",
            color="blue",
            linestyle="--",
            label=f"{metric_name}",
        )
        axes[i, j].legend(loc="upper right", fontsize="small", framealpha=0.5)
        axes[i, j].grid(visible=True)

plt.subplots_adjust(hspace=0.5)
plt.savefig("./train_images/tuning_results/result_5-10_units", dpi=300)
