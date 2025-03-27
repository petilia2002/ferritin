import pandas as pd

df = pd.read_csv("./data/lab-2025.csv", sep=",", dtype={"sampleno": str, "hgb": float})
print(df.shape)
print(df)
print(df.dtypes)
