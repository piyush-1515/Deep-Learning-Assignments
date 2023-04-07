#!/usr/bin/env python
# coding: utf-8

# In[8]:


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
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory


# In[19]:


print(tf.__version__)


# In[9]:


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


# In[10]:


def normalize(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image, label

train = train.map(normalize)
val = val.map(normalize)
test = test.map(normalize)


# In[11]:


# model architectures - 3 hidden layers
num_hidden = 3
model_arch = [
    [4, 4, 4],
    [4, 8, 16],
    [16, 8, 4],
    [8, 8, 8],
    [4, 16, 32],
    [32, 16, 4],
    [16, 16, 16],
    [16, 32, 64],
    [64, 32, 16],
    [32, 32, 32],
    [32, 64, 96],
    [96, 64, 32],
    [64, 64, 64],
    [64, 96, 128],
    [128, 96, 64],
    [128, 128, 128],
]


# In[5]:


# optimizers
sgd_optimizer = SGD(learning_rate=0.001,name='SGD') #Stochastic Gradient Descent ##
batch_optimizer = SGD(learning_rate=0.001,name='batch') #Batch Gradient Descent ##
momentum_optimizer = SGD(learning_rate=0.001, momentum=0.9, name='Momentum_SGD') #Momentum Based
nag_optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True, name='NAG') #NAG
rms_optimizer = RMSprop(learning_rate=0.001, rho=0.99, momentum=0.0, epsilon=1e-8, name="RMSProp") #RMSProp
adagrad_optimizer = Adagrad(learning_rate=0.001, epsilon=1e-8, name="Adagrad") #AdaGrad
adam_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) #Adam

optimizers = [
    ['sgd', sgd_optimizer],
    ['batch', batch_optimizer], ##
    ['momentum', momentum_optimizer],
    ['nag', nag_optimizer],
    ['rmsprop', rms_optimizer],
    ['adagrad', adagrad_optimizer],
    ['adam', adam_optimizer]
]


# In[6]:


# kernel initializer
initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)


# In[22]:


k=5 # no. of classes
input_shape = (28, 28, 1)

model_history = dict()

# train different achitectures and optimizers
print('Training models with different architectures and optimizers')
for layer_dims in [model_arch[1]]:
    for optimizer in optimizers:
        print(f'{optimizer[0]}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}...')
        # define model
        model = Sequential([
            keras.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(layer_dims[0], activation="sigmoid", name="layer1", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
            layers.Dense(layer_dims[1], activation="sigmoid", name="layer2", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
            layers.Dense(layer_dims[2], activation="sigmoid", name="layer3", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
            layers.Dense(k, activation="softmax", name="output", 
                         kernel_initializer=initializer, bias_initializer=initializers.Zeros()),
        ])
        
        # compile model
        model.compile(optimizer=optimizer[1], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # callbacks
        my_callbacks = [
            EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3),
            TensorBoard(log_dir=f'./logdir/{optimizer[0]}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}/')
        ]
        
        batch_size=1
        if optimizer[0]=='batch':
            batch_size = train.cardinality().numpy()
        
        model_fit = model.fit(train, batch_size=batch_size, epochs=10000, verbose=0, callbacks=my_callbacks, 
                              validation_split=0.0, validation_data=val, shuffle=True, validation_batch_size=None)
        
        model_history[f'{optimizer[0]}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}'] = model_fit.history['accuracy']
        
        hist_metric = 'accuracy'
        print(f'epochs: {len(model_fit.history[hist_metric])}, acc: {model_fit.history[hist_metric][-1]}\n')
        model.save(f'models/{optimizer[0]}-{layer_dims[0]}-{layer_dims[1]}-{layer_dims[2]}.tf')


# In[15]:


import pickle
hist = open('history2', 'wb')
pickle.dump(model_history, hist)
hist.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




