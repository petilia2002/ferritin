import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# Пример данных
x = np.random.randn(1000)

# Создание объекта KDE
kde = gaussian_kde(x)

# Создание массива точек для оценки плотности
x_vals = np.linspace(x.min() - 1, x.max() + 1, 500)

# Оценка плотности в этих точках
y_vals = kde(x_vals)

# Поиск максимума плотности
max_density = np.max(y_vals)
max_density_point = x_vals[np.argmax(y_vals)]

# Визуализация
plt.plot(x_vals, y_vals, label="KDE")

# Отображаем точку максимума плотности как жирный маркер
plt.scatter(
    max_density_point,
    max_density,
    color="red",
    s=20,
    zorder=5,
    label=f"Максимум: {max_density_point:.2f}",
)

# Дополнительные элементы графика
plt.fill_between(x_vals, y_vals, alpha=0.3)
plt.axvline(max_density_point, color="r", linestyle="--")
plt.xlabel("x")
plt.ylabel("Плотность")
plt.title("Оценка плотности распределения и её максимум")
plt.legend(loc="upper right")
plt.show()
