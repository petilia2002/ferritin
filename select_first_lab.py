import pandas as pd

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
print(lab.shape)
print(lab)

ferritin = (
    lab[(lab["ferritin"].notna()) & (lab["ferritin"] >= 0.0) & (lab["lab_id"] == 1)]
    .reset_index(drop=True)
    .filter(items=labels)
)
print(ferritin.shape)
print(ferritin)

ferritin.dropna(inplace=True, ignore_index=True)
print(ferritin.shape)
print(ferritin)

ferritin.to_csv("./data/ferritin-lab1.csv", sep=",", index=False)
