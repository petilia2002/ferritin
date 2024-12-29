import json

metrics_by_units = [
    {"units": 20, "AUC": 84},
    {"units": 30, "AUC": 85},
    {"units": 50, "AUC": 87},
]
print(metrics_by_units)

with open("./output/metrics_by_units.json", "w", encoding="utf-8") as file:
    json.dump(metrics_by_units, file, ensure_ascii=False, indent=4)

import json
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные из JSON
data = [
    {"units": 20, "AUC": 84},
    {"units": 30, "AUC": 87},
    {"units": 50, "AUC": 86},
    {"units": 70, "AUC": 86.5},
]

# Преобразуем данные в списки для построения графика
units = [item["units"] for item in data]
auc = [item["AUC"] for item in data]

# Построение графика
plt.figure(figsize=(8, 6))
sns.lineplot(x=units, y=auc, marker="o", color="b", label="AUC", linestyle="--")
plt.title("Зависимость AUC от количества нейронов", fontsize=16)
plt.xlabel("Количество нейронов (units)", fontsize=14)
plt.ylabel("Метрика AUC", fontsize=14)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
