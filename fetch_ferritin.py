from configparser import ConfigParser
import fdb
import pandas as pd

config_object = ConfigParser()
config_object.read("./config.ini")

user_info = config_object["USERINFO"]
server_config = config_object["SERVERCONFIG"]

host = server_config["host"]
database = server_config["database"]
charset = server_config["charset"]
user = user_info["user"]
password = user_info["password"]

connection = fdb.connect(
    host=host, database=database, user=user, password=password, charset=charset
)

print(f"Successfully connected to {database} database")

rows_count = 10000
select = f"select * from lab where lab.ferritin > 0 rows {rows_count}"

cursor = connection.cursor()
cursor.execute(select)
rows = cursor.fetchall()

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

targets = ["ferritin"]

columns = [t[0] for t in cursor.description]
columns = list(map(str.lower, columns))
print(columns)

df = pd.DataFrame(data=rows, columns=columns)
df = df.filter(items=labels)

df[
    [
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
] = df[
    [
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
].astype(
    "float"
)

print(df.head())
print(df.shape)
print(df.dtypes)

df.to_csv("data/ferritin.csv", sep=",", index=False)
connection.close()
print(f"Closed {database} database")
