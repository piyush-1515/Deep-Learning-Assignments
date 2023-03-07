# %%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from fcnn import *
data = pd.read_csv(r'Group24\Group24\Regression\UnivariateData\24.csv', names=['x', 'y'])
X, Y = data.x, data.y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
# %%

FCNN = Model(0.01,"regression")
nodes_hidden_layer = 15
FCNN.add_layer(1)
FCNN.add_layer(nodes_hidden_layer)
# FCNN.add_layer(nodes_hidden_layer)
FCNN.add_layer(1)

errors = []
previous_validation_error = 100
inc_val_er_streak = 0
while True:
    FCNN.fit(X_train.to_numpy().reshape(len(X_train),1), y_train.to_numpy().reshape(len(y_train),1))
    errors.append(FCNN.avg_training_error())
    if((len(errors)>1 and abs(errors[len(errors)-1]-errors[len(errors)-2])<0.00001) or len(errors)>500): break
    
    # Computing the validation error
    prediction = FCNN.classify_batch(X_val.to_numpy().reshape(len(X_val),1))
    current_validation_error = mean_squared_error(y_val, prediction)
    if(previous_validation_error < current_validation_error): 
        inc_val_er_streak += 1
        if(inc_val_er_streak >= 3): break
    inc_val_er_streak = 0        
    previous_validation_error = current_validation_error
    FCNN.total_error = []

# %%
# Epoch vs Avergae error
plt.figure(figsize=(10,5))
plt.plot([i for i in range(1,len(errors)+1)],errors)
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.title("FIg 1. Epoch V/S Average error")
plt.grid(True)
plt.show()

# %%
val_mse = 0
# Classification of validation data
predicted_val = FCNN.classify_batch(X_val.to_numpy().reshape(len(X_val),1))

# Mean squared error
val_mse = mean_squared_error(y_val, predicted_val)
print("MSE (Validation data) = ", val_mse)

# %%
train_mse = 0
# Classification of training data
predicted_train = FCNN.classify_batch(X_train.to_numpy().reshape(len(X_train),1))

# Mean squared error
train_mse = mean_squared_error(y_train, predicted_train)
print("MSE (training data) = ", train_mse)

# %%
test_mse = 0
# Classification of testing data
predicted_test = FCNN.classify_batch(X_test.to_numpy().reshape(len(X_test),1))

# Mean squared error
test_mse = mean_squared_error(y_test, predicted_test)
print("MSE (testing data) = ", test_mse)

# %%

# Model output and target output for validation data
plt.scatter(X_val, y_val)
plt.scatter(X_val, predicted_val)
plt.xlabel("Input (x)")
plt.ylabel("Predicted Output")
plt.title("Model output and target output for validation data")
plt.legend(['Model Output', 'Target Output'])
plt.show()
# %%

# Target output vs model output for validation data
plt.scatter(y_val, predicted_val)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Validation)")
plt.show()
# %%

# Model output and target output for training data
plt.scatter(X_train, y_train)
plt.scatter(X_train, predicted_train)
plt.xlabel("Input (x)")
plt.ylabel("Predicted Output")
plt.title("Model output and target output for training data")
plt.legend(['Model Output', 'Target Output'])
plt.show()
# %%

# Target output vs model output for training data
plt.scatter(y_train, predicted_train)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Training)")
plt.show()

# %%
# Model output and target output for testing data
plt.scatter(X_test, y_test)
plt.scatter(X_test, predicted_test)
plt.xlabel("Input (x)")
plt.ylabel("Predicted Output (x)")
plt.title("Model output and target output for testing data")
plt.legend(['Model Output', 'Target Output'])
plt.show()

# %%

# Target output vs model output for testing data
plt.scatter(y_test, predicted_test)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Testing)")
plt.show()

# %%

plt.bar(['Training', 'Validation', 'Testing'], [train_mse, val_mse, test_mse], color='maroon', width=0.4)
plt.title("Training and testing MSE")
plt.ylabel("MSE values")
plt.show()

#%%


fig = plt.figure(figsize=(24, 28))
fig.suptitle('Fig 5. Hidden Layer (Validation Set)', y=0.48, fontsize=24)
for l in range(1,8+1): #choosing only first 8 neurons
    neuron_activation_value = []
    for i in range(X_val.shape[0]):
        neuron_activation_value.append(FCNN.classify_point([X_val.to_numpy()[i]])[1][1].gn[l])
        FCNN.clean_layers()

    plt.subplot(4, 4, l)
    # ax = fig.add_subplot(2, 3, l, projection='3d', frame_on=True)
    plt.scatter(X_val, neuron_activation_value)

    plt.xlabel("Input (x)", fontsize=15)
    plt.ylabel("Neuron Activation", fontsize=15)
    plt.title("{}) Neuron {}".format(chr(96+l),l),y=-0.18, fontsize=15)
    plt.grid(True)

plt.show()

#%%


fig = plt.figure(figsize=(24, 28))
fig.suptitle('Fig 6. Hidden Layer (Training Set)', y=0.48, fontsize=24)
for l in range(1,8+1): #choosing only first 8 neurons
    neuron_activation_value = []
    for i in range(X_train.shape[0]):
        neuron_activation_value.append(FCNN.classify_point([X_train.to_numpy()[i]])[1][1].gn[l])
        FCNN.clean_layers()

    plt.subplot(4, 4, l)
    # ax = fig.add_subplot(2, 3, l, projection='3d', frame_on=True)
    plt.scatter(X_train, neuron_activation_value)

    plt.xlabel("Input (x)", fontsize=15)
    plt.ylabel("Neuron Activation", fontsize=15)
    plt.title("{}) Neuron {}".format(chr(96+l),l),y=-0.18, fontsize=15)
    plt.grid(True)

plt.show()

#%%


fig = plt.figure(figsize=(24, 28))
fig.suptitle('Fig 7. Hidden Layer (Test Set)', y=0.48, fontsize=24)
for l in range(1,nodes_hidden_layer+1): #choosing only first 8 neurons
    neuron_activation_value = []
    for i in range(X_test.shape[0]):
        neuron_activation_value.append(FCNN.classify_point([X_test.to_numpy()[i]])[1][1].gn[l])
        FCNN.clean_layers()

    plt.subplot(4, 4, l)
    # ax = fig.add_subplot(2, 3, l, projection='3d', frame_on=True)
    plt.scatter(X_test, neuron_activation_value)

    plt.xlabel("Input (x)", fontsize=15)
    plt.ylabel("Neuron Activation", fontsize=15)
    plt.title("{}) Neuron {}".format(chr(96+l),l),y=-0.18, fontsize=15)
    plt.grid(True)

plt.show()

#%%

# Best model: for 1 layer
# eta = 0.01
# stopping : error diff = 0.00001
# hidden layer1 : 15

# Best model: for 2 layer
# eta = 0.01
# stopping : error diff = 0.00001
# hidden layer1 : 15
# hidden layer2 : 15

# hnodes = 15
# MSE (training data) =  0.007069919324998826