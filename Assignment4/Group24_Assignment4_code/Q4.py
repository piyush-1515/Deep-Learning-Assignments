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
# #### Finding compressed representation

# %%
def get_reduced_representation(dim, vec):
    models = os.listdir('./models/Q2')
    model_path = str
    for model in models:
        if( ("3_hidden" in model) and ((str(dim)+"_n") in model)) : model_path = model
    loaded_model = tf.keras.models.load_model(os.path.join('./models/Q2', model_path))
    hidden_layer_model = tf.keras.models.Model(inputs=loaded_model.input, outputs=loaded_model.layers[2].output)
    hidden_output = hidden_layer_model.predict(vec[0].numpy().reshape(len(vec[0]),784))
    return hidden_output

# %% [markdown]
# #### Building classifiers

# %%
model_arch = [
    [96, 64, 32],
    [64, 96, 128],
    [128, 96, 64],
    [256, 128, 96]
]
initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)

k=5 # no. of classes
model_history = dict()

# train different achitectures and optimizers
print('Training models with different architectures')
for reduced_dimension in [32,64,128,256]:
    reduced_rep_train = get_reduced_representation(reduced_dimension, train_vectors)
    reduced_rep_val = get_reduced_representation(reduced_dimension, val_vectors)
    for layer_dims in model_arch:
        print(f'{reduced_dimension}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}...')
        # define model
        model = Sequential([
            layers.Dense(reduced_dimension, activation="relu", input_shape=(reduced_dimension,)),
            # keras.Input(input_shape=(reduced_dimension,)),
            layers.Dense(layer_dims[0], activation="sigmoid", name="layer1", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
            layers.Dense(layer_dims[1], activation="sigmoid", name="layer2", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
            layers.Dense(layer_dims[2], activation="sigmoid", name="layer3", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
            # layers.Dense(layer_dims[3], activation="sigmoid", name="layer4", 
            #              kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
            layers.Dense(k, activation="softmax", name="output", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        ])
        
        # compile model
        adam_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # callbacks
        my_callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=10),
            TensorBoard(log_dir=f'./logdir/Q3/{reduced_dimension}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}/')
        ]
        model_fit = model.fit(reduced_rep_train, train_vectors[1].numpy(), batch_size=len(train_vectors[0]), epochs=10000, verbose=0, callbacks=my_callbacks, 
                              validation_split=0.0, validation_data=(reduced_rep_val, val_vectors[1].numpy()), shuffle=True, validation_batch_size=None)
        
        model_history[f'{reduced_dimension}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}'] = model_fit.history['accuracy']
        
        hist_metric = 'accuracy'
        print(f'epochs: {len(model_fit.history[hist_metric])}, acc: {model_fit.history[hist_metric][-1]}\n')
        model.save(f'models/Q3/{reduced_dimension}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}.tf')




