# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import initializers
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

# %% [markdown]
# #### Plotting the weights of single hidden layer denoising autoencoder

# %%
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
weights = tf.keras.models.load_model("./models/Q5/Denoising_20per_32_neurons.tf/", custom_objects={'custom_loss': custom_loss}).get_weights()[2]
fig , ax = plt.subplots(8, 4)
fig.set_size_inches(20,40)
ct = 0
for i in range(0, 32):
    ax[int(i/4)][int(i%4)].imshow(weights[i].reshape(28,28))
plt.show()

# %% [markdown]
# #### Plotting the weights of single hidden layer autoencoder

# %%
weights = tf.keras.models.load_model("./models/Q2/1_hidden_layer_32_neurons.tf/").get_weights()[4]
fig , ax = plt.subplots(8, 4)
fig.set_size_inches(20,40)
ct = 0
for i in range(0, 32):
    ax[int(i/4)][int(i%4)].imshow(weights[i].reshape(28,28))
plt.show()


