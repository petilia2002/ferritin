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
print(lab.head())
print(lab.shape)

ferritin = (
    lab[(lab["ferritin"].notna()) & (lab["ferritin"] > 0)]
    .reset_index(drop=True)
    .filter(items=labels)
)
print(ferritin.head())
print(ferritin.shape)

z_score = 1.96
conf_intervals = {}
for label in labels:
    if label == "gender":
        continue
    conf_intervals[label] = {}
    data = ferritin[label]
    mean_value = data.mean()
    std_value = data.std()
    low, high = (
        mean_value - z_score * std_value,
        mean_value + z_score * std_value,
    )
    conf_intervals[label]["low"] = low
    conf_intervals[label]["high"] = high

for label in labels:
    if label == "gender":
        continue
    low, high = conf_intervals[label]["low"], conf_intervals[label]["high"]
    ferritin = ferritin[(ferritin[label] >= low) & (ferritin[label] <= high)]

ferritin.reset_index(drop=True, inplace=True)
print(ferritin.head())
print(ferritin.shape)
ferritin.to_csv("./data/ferritin-72k.csv", sep=",", index=False)

columns_to_check = ferritin.drop(columns=["gender"]).columns

filtered_df = ferritin[(ferritin[columns_to_check] <= 0).any(axis=1)]
print(filtered_df.head())
print(filtered_df.shape)
