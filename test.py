import pandas as pd

df = pd.read_csv("./data/ferritin-72k.csv", sep=",", dtype={"hgb": float})
print(df.head())
print(df.shape)
print(df.dtypes)
