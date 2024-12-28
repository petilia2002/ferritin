from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 10)
y = np.arange(0, 10)

# Строим график
plt.plot(x, y, color="blue", linewidth=2)
plt.title("График функции")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.show()

print(integrate.simpson(y, x=x))
