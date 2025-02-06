import numpy as np

x = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([7, 8, 11])]
y = [2 * np.array([[1, 2, 3], [4, 5, 6]]), 2 * np.array([7, 8, 11])]
w = []
w.append(x)
w.append(y)
# print(w)

# # Реализация:
# weights = list(map(lambda x: x[0], w))
# biases = list(map(lambda x: x[1], w))

# new_weights = [np.mean(weights, axis=0), np.mean(biases, axis=0)]

# print()
# print(new_weights)

# average_weights = []
# for weights in zip(*w):
#     average_weights.append(np.mean(weights, axis=0))

# print(average_weights)

a = [1, 2, 3, 4, 5]
print(a[-4:])
