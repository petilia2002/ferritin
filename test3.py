user = {"name": "Ilya", "age": 22, "avg_scores": 90}

keys = [key for key in user.keys() if key.startswith("avg_")]
print(keys)
