import json
import pandas as pd

with open("./output/labs.json", "r", encoding="utf-8") as file:
    statistic = json.load(file)

rows = []
for measure, labs in statistic.items():
    for lab_id, stats in labs.items():
        row = {"measure": measure, "lab_id": lab_id}
        row.update(stats)
        rows.append(row)

df = pd.DataFrame(rows)
df = df[["measure", "lab_id", "min", "max", "mean", "std"]]

table = df.to_string(index=False)
print(table)

with open("./output/labs.txt", "w", encoding="utf-8") as file:
    file.write(table)
