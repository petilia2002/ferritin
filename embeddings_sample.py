from keras import Model
from keras.api.layers import Input, Embedding
import numpy as np

gpt = np.load("./assets/embeddings.npy")
dims = gpt.shape

l_in = Input(name="input_analytes", shape=(None,))
l_embed = Embedding(name="embedding", input_dim=dims[0], output_dim=dims[1])(l_in)

embed = Model(l_in, l_embed)
embed.set_weights([gpt])
embed.trainable = False

embed.summary()

x = np.array([[0, 2, 34, 36]])
y = embed(x)

print(y)

print(embed.get_weights()[0][[0, 2, 34, 36]])
print(embed.get_weights()[0].shape)
