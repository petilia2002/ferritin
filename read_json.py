import json
import pandas as pd

with open("./output/labs.json", "r", encoding="utf-8") as file:
    statistic = json.load(file)

rows = []
for measure, stats in statistic.items():
    for lab, estimations in stats.items():
        row = {"measure": measure, "lab_id": lab}
        row.update(estimations)
        rows.append(row)

df = pd.DataFrame(rows)
txt_string = df.to_string(index=True)

with open("./output/labs.txt", "w", encoding="utf-8") as file:
    file.write(txt_string)
