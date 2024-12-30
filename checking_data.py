import pandas as pd

df = pd.read_csv("./data/ferritin-v2.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)

columns_to_check = df.drop(columns=["gender"]).columns

filtered_df = df[(df[columns_to_check] == 0).any(axis=1)].reset_index(drop=True)
print(filtered_df.head())
print(filtered_df.shape)
