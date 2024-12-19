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

select = "select * from lab"
cursor = connection.cursor()

cursor.execute(select)
rows = cursor.fetchall()

columns = [t[0] for t in cursor.description]
columns = list(map(str.lower, columns))

df = pd.DataFrame(data=rows, columns=columns)
print(df.head())
print(df.shape)
print(df.dtypes)

df.to_csv("data/lab.csv", sep=",", index=False)
connection.close()
print(f"Closed {database} database")
