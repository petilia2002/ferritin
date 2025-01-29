import keras
from keras import layers
import tensorflow as tf

# Размерности
height, width = 5, 5

# Создание модели
input_tensor = layers.Input(
    shape=(height, width)
)  # Входной тензор (batch_size, height, width)

# Создание булевой маски (где True сохраняет значение, а False — исключает)
mask = layers.Input(
    shape=(height, width), dtype=tf.bool
)  # Маска (batch_size, height, width)

# Применение маски с использованием tf.boolean_mask
masked_input = tf.boolean_mask(
    input_tensor, mask, axis=[1, 2]
)  # Отсеиваем элементы, где маска False

# Сплющивание итогового массива, так как результат после mask — это одномерный тензор
flattened = layers.Flatten()(masked_input)

# Если нужно, можно добавить слои после flatten:
output = layers.Dense(10)(flattened)

model = keras.Model(inputs=[input_tensor, mask], outputs=output)

# Компиляция модели
model.compile(optimizer="adam", loss="mse")
