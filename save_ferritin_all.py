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
    lab[(lab["ferritin"].notna()) & (lab["ferritin"] >= 0.0)]
    .reset_index(drop=True)
    .filter(items=labels)
)
print(ferritin.shape)
print(ferritin)

ferritin.dropna(inplace=True, ignore_index=True)
print(ferritin.shape)
print(ferritin)

# temp_df = ferritin.drop(columns=["gender"], inplace=False)
# res = (temp_df[temp_df.columns] < 0.0).any(axis="columns").any(axis="index")
# print(res.shape)
# print(res)


# columns_to_check = temp_df.columns
# ferritin = ferritin[(ferritin[columns_to_check] < 0.0).any(axis="columns")].reset_index(
#     drop=True
# )
# print(ferritin.shape)
# print(ferritin)

ferritin.to_csv("./data/ferritin-all.csv", sep=",", index=False)
