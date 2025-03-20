from configparser import ConfigParser
import numpy as np
from yandex_cloud_ml_sdk import YCloudML

config_object = ConfigParser()
config_object.read("./config.ini")
folder_id = config_object["YANDEX"]["folder_id"]
auth = config_object["YANDEX"]["auth"]

sdk = YCloudML(
    folder_id=folder_id,
    auth=auth,
)

path = "./assets/my-embeddings.npy"

idx = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    30,
    31,
    32,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    34,
    35,
    36,
    37,
]
analytes = [
    "ферритин, нг/мл",
    "витамин В12, пг/мл",
    "фолиевая кислота, нг/мл",
    "аспартатаминотрансфераза, ед/л",
    "аланинаминотрансфераза, ед/л",
    "билирубин прямой, мкмоль/л",
    "билирубин непрямой, мкмоль/л",
    "билирубин общий, мкмоль/л",
    "креатинин, мкмоль/л",
    "мочевина, ммоль/л",
    "белок общий, г/л",
    "лактатдегидрогеназа, ед/л",
    "холестерин, ммоль/л",
    "глюкоза, ммоль/л",
    "мочевая кислота, ммоль/л",
    "альбумин, г/л",
    "гемоглобин, г/л",
    "эритроциты, 10^12/л",
    "средний объем эритроцитов, фл",
    "лейкоциты, 10^9/л",
    "тромбоциты, 10^9/л",
    "нейтрофилы, %",
    "лимфоциты, %",
    "эозинофилы, %",
    "базофилы, %",
    "моноциты, %",
    "С-реактивный белок, мг/л",
    "пол",
    "возраст",
    "средние клетки, %",
    "гранулоциты, %",
]

query_model = sdk.models.text_embeddings("query")
embeddings = np.zeros((38, 256))

for i, analyte in enumerate(analytes):
    query_embedding = query_model.run(analyte)
    embeddings[idx[i], :] = np.array(query_embedding)

print(embeddings.shape)
print(embeddings)
np.save(path, embeddings)
print(f"Файл сохранен")

loaded_arr = np.load(path)
print((loaded_arr == embeddings).all())
