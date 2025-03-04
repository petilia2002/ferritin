import numpy as np
import pandas as pd

df = pd.read_csv(
    "./data/ferritin-all.csv", sep=",", dtype={"sampleno": str, "hgb": float}
)
print(df.shape)
print(df)

hgb, rbc = df["hgb"], df["rbc"]
df["mch"] = round(hgb / rbc, 2)

mcv = df["mcv"]
df["hct"] = round(mcv * rbc / 10, 2)
df["mchc"] = round(hgb * 10 / df["hct"], 2)
df["cp"] = round(hgb * 3.0 / (rbc * 100.0), 2)
print(df.shape)
print(df)

df_cleaned = df.replace([np.inf, -np.inf], np.nan)
df_cleaned.dropna(inplace=True, ignore_index=True)
print(df_cleaned.shape)
print(df_cleaned)

df_cleaned.to_csv("./data/ferritin-all-calc.csv", sep=",", index=False)
