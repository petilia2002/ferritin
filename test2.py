import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

df = pd.read_csv("./data/lab.csv", sep=",", dtype={"sampleno": str, "hgb": float})
# df = df.filter(items=labels)
print(df.info())
df.dropna(inplace=True, ignore_index=True)
print(df.info())
