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
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import initializers
from keras.optimizers import Adam
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt

# %% [markdown]
# #### Reading the data, normalizing and flattening it

# %% [markdown]
# ##### Reading and Normalizing

# %%

train = image_dataset_from_directory(
    'Group_24/train/',
    labels='inferred',
    label_mode='int',
    batch_size=1,
    image_size=(28, 28),
    shuffle=True,
    seed=42,
    color_mode='grayscale',
    validation_split=0.0
)

val = image_dataset_from_directory(
    'Group_24/val/',
    labels='inferred',
    label_mode='int',
    batch_size=1,
    image_size=(28, 28),
    shuffle=True,
    seed=42,
    color_mode='grayscale',
    validation_split=0.0
)

test = image_dataset_from_directory(
    'Group_24/test/',
    labels='inferred',
    label_mode='int',
    batch_size=1,
    image_size=(28, 28),
    shuffle=True,
    seed=42,
    color_mode='grayscale',
    validation_split=0.0
)

def normalize(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label

train = train.map(normalize)
val = val.map(normalize)
test = test.map(normalize)

# %% [markdown]
# ##### Flattening

# %% [markdown]
# Preparing training tensors

# %%
# Iterate through the dataset and reshape each image tensor
image_tensors = []
label_tensors = []
for image, labels in train:
    num_images = image.shape[0]
    image_vectors = tf.reshape(image, [num_images, -1])
    image_tensors.append(image_vectors)
    label_tensors.append(labels)

# Concatenate the image tensors into a single tensor
train_vectors = [tf.concat(image_tensors, axis=0), tf.concat(label_tensors, axis=0)]

# %% [markdown]
# Preparing validation tensors

# %%
# Iterate through the dataset and reshape each image tensor
image_tensors = []
label_tensors = []
for image, labels in val:
    num_images = image.shape[0]
    image_vectors = tf.reshape(image, [num_images, -1])
    image_tensors.append(image_vectors)
    label_tensors.append(labels)

# Concatenate the image tensors into a single tensor
val_vectors = [tf.concat(image_tensors, axis=0), tf.concat(label_tensors, axis=0)]

# %% [markdown]
# Preparing testing tensors

# %%
# Iterate through the dataset and reshape each image tensor
image_tensors = []
label_tensors = []
for image, labels in test:
    num_images = image.shape[0]
    image_vectors = tf.reshape(image, [num_images, -1])
    image_tensors.append(image_vectors)
    label_tensors.append(labels)

# Concatenate the image tensors into a single tensor
test_vectors = [tf.concat(image_tensors, axis=0), tf.concat(label_tensors, axis=0)]

# %% [markdown]
# #### 1 Hidden layer autoencoder

# %%
initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)

model_history = dict()

print('Training 1 Hidden Layer autoencoder with different number of neurons in bottleneck layer')
for num_neurons in [32, 64, 128, 256]:
    print(f'1 Hidden Layer : {num_neurons} neurons')
    # define model
    model = Sequential([
        layers.Dense(784, activation="relu", input_shape=(784,)),
        layers.Dense(num_neurons, activation="sigmoid", name="hidden", 
                        kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        layers.Dense(784, activation="relu", name="output", 
                        kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
    ])
    
    # compile model
    loss = tf.keras.losses.MeanSquaredError()
    adam_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam_optimizer, loss=loss, metrics=['mse'])
    
    # callbacks
    my_callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5),
        TensorBoard(log_dir=f'./logdir/Q2/1_hidden_layer_{num_neurons}_neurons/')
    ]

    model_fit = model.fit(train_vectors[0].numpy(),train_vectors[0].numpy(), batch_size=len(train_vectors[0]), epochs=10000, verbose=0, callbacks=my_callbacks, 
                            validation_split=0.0, validation_data=(val_vectors[0].numpy(), val_vectors[0].numpy()), shuffle=True, validation_batch_size=None)
    
    model_history[f'1_hidden_layer_{num_neurons}_neurons'] = model_fit.history['mse']
    
    hist_metric = 'mse'
    print(f'epochs: {len(model_fit.history[hist_metric])}, mse: {model_fit.history[hist_metric][-1]}\n')
    model.save(f'models/Q2/1_hidden_layer_{num_neurons}_neurons.tf')



# %% [markdown]
# ##### Displaying reconstructed image for training data

# %%
plt.imshow(train_vectors[0].numpy()[0].reshape(28,28))

# %%
fig, ax = plt.subplots(nrows=2, ncols=2)
ct = 0
for model in os.listdir("./models/Q2"):
    # models = os.listdir('./models/Q1')
    model_path = os.path.join('./models/Q2', model)
    if(model[0]!='1') : break
    print(model)
    trained_model = tf.keras.models.load_model(model_path)
    predictions = trained_model.predict(train_vectors[0].numpy()[0].reshape(1,784))
    ax[int(ct/2)][int(ct%2)].imshow(predictions.reshape(28,28))
    ct += 1

# %% [markdown]
# ##### Displaying reconstructed images for validation data

# %%
plt.imshow(val_vectors[0].numpy()[0].reshape(28,28))

# %%
fig, ax = plt.subplots(nrows=2, ncols=2)
ct = 0
for model in os.listdir("./models/Q2"):
    # models = os.listdir('./models/Q1')
    model_path = os.path.join('./models/Q2', model)
    if(model[0]!='1') : break
    print(model)
    trained_model = tf.keras.models.load_model(model_path)
    predictions = trained_model.predict(val_vectors[0].numpy()[0].reshape(1,784))
    ax[int(ct/2)][int(ct%2)].imshow(predictions.reshape(28,28))
    ct += 1

# %% [markdown]
# ##### Displaying reconstructed images for testing data

# %%
plt.imshow(test_vectors[0].numpy()[0].reshape(28,28))

# %%
fig, ax = plt.subplots(nrows=2, ncols=2)
ct = 0
for model in os.listdir("./models/Q2"):
    # models = os.listdir('./models/Q1')
    model_path = os.path.join('./models/Q2', model)
    if(model[0]!='1') : break
    print(model)
    trained_model = tf.keras.models.load_model(model_path)
    predictions = trained_model.predict(test_vectors[0].numpy()[0].reshape(1,784))
    ax[int(ct/2)][int(ct%2)].imshow(predictions.reshape(28,28))
    ct += 1

# %% [markdown]
# #### 3 Hidden layer autoencoder

# %%
initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)

model_history = dict()

print('Training 3 Hidden Layer autoencoder with different number of neurons in bottleneck layer')
for num_neurons in [32, 64, 128, 256]:
    print(f'1 Hidden Layer : {num_neurons} neurons')
    # define model
    model = Sequential([
        layers.Dense(784, activation="relu", input_shape=(784,)),
        layers.Dense(400, activation="sigmoid", name="hidden1", 
                        kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        layers.Dense(num_neurons, activation="sigmoid", name="bottleneck", 
                        kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        layers.Dense(400, activation="sigmoid", name="hidden3", 
                        kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        layers.Dense(784, activation="relu", name="output", 
                        kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
    ])
    
    # compile model
    loss = tf.keras.losses.MeanSquaredError()
    adam_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam_optimizer, loss=loss, metrics=['mse'])
    
    # callbacks
    my_callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5),
        TensorBoard(log_dir=f'./logdir/Q2/3_hidden_layer_{num_neurons}_neurons/')
    ]

    model_fit = model.fit(train_vectors[0].numpy(),train_vectors[0].numpy(), batch_size=len(train_vectors[0]), epochs=10000, verbose=0, callbacks=my_callbacks, 
                            validation_split=0.0, validation_data=(val_vectors[0].numpy(), val_vectors[0].numpy()), shuffle=True, validation_batch_size=None)
    
    model_history[f'3_hidden_layer_{num_neurons}_neurons'] = model_fit.history['mse']
    
    hist_metric = 'mse'
    print(f'epochs: {len(model_fit.history[hist_metric])}, mse: {model_fit.history[hist_metric][-1]}\n')
    model.save(f'models/Q2/3_hidden_layer_{num_neurons}_neurons.tf')



# %% [markdown]
# ##### Displaying reconstructed image for training data

# %%
plt.imshow(train_vectors[0].numpy()[0].reshape(28,28))
fig, ax = plt.subplots(nrows=2, ncols=2)
ct = 0
for model in os.listdir("./models/Q2"):
    # models = os.listdir('./models/Q1')
    model_path = os.path.join('./models/Q2', model)
    if(model[0]!='3') : break
    print(model)
    trained_model = tf.keras.models.load_model(model_path)
    predictions = trained_model.predict(train_vectors[0].numpy()[0].reshape(1,784))
    ax[int(ct/2)][int(ct%2)].imshow(predictions.reshape(28,28))
    ct += 1

# %% [markdown]
# ##### Displaying reconstructed image for validation data

# %%
plt.imshow(val_vectors[0].numpy()[0].reshape(28,28))
fig, ax = plt.subplots(nrows=2, ncols=2)
ct = 0
for model in os.listdir("./models/Q2"):
    # models = os.listdir('./models/Q1')
    model_path = os.path.join('./models/Q2', model)
    if(model[0]!='3') : break
    print(model)
    trained_model = tf.keras.models.load_model(model_path)
    predictions = trained_model.predict(val_vectors[0].numpy()[0].reshape(1,784))
    ax[int(ct/2)][int(ct%2)].imshow(predictions.reshape(28,28))
    ct += 1

# %% [markdown]
# ##### Displaying reconstructed image for testing data

# %%
plt.imshow(test_vectors[0].numpy()[0].reshape(28,28))
fig, ax = plt.subplots(nrows=2, ncols=2)
ct = 0
for model in os.listdir("./models/Q2"):
    # models = os.listdir('./models/Q1')
    model_path = os.path.join('./models/Q2', model)
    if(model[0]!='3') : break
    print(model)
    trained_model = tf.keras.models.load_model(model_path)
    predictions = trained_model.predict(test_vectors[0].numpy()[0].reshape(1,784))
    ax[int(ct/2)][int(ct%2)].imshow(predictions.reshape(28,28))
    ct += 1


