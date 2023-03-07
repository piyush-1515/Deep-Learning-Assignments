# %%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from fcnn import *

data = pd.read_csv(r'Group24\Group24\Regression\BivariateData\24.csv', names=['x', 'y', 'z'])
target_output = data.z
data = data.drop(columns=['z'])

X_train, X_test, y_train, y_test = train_test_split(data, target_output, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# %%

FCNN = Model(0.1,"regression")
nodes_hidden_layer1 = 6
nodes_hidden_layer2 = 6
FCNN.add_layer(2)
FCNN.add_layer(nodes_hidden_layer1)
# FCNN.add_layer(nodes_hidden_layer2)
FCNN.add_layer(1)

errors = []
previous_validation_error = 100
inc_val_er_streak = 0
while True:
    FCNN.fit(X_train.to_numpy(), y_train.to_numpy().reshape(len(y_train),1))
    errors.append(FCNN.avg_training_error())
    if(len(errors)>1 and abs(errors[len(errors)-1]-errors[len(errors)-2])<0.0001): break

    # Computing the validation error
    prediction = FCNN.classify_batch(X_val.to_numpy())
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
plt.title("Fig 8. Epoch V/S Average error",y=-0.2)
plt.grid(True)
plt.show()

# %%
# Classification of validation data
val_mse = 0
predicted_val = FCNN.classify_batch(X_val.to_numpy())

# Mean squared error
val_mse = mean_squared_error(y_val, predicted_val)
print("MSE (Validation data) = ", val_mse)

# %%
# Classification of training data
train_mse = 0
predicted_train = FCNN.classify_batch(X_train.to_numpy())

# Mean squared error
train_mse = mean_squared_error(y_train, predicted_train)
print("MSE (training data) = ", train_mse)

#%%
test_mse = 0
# Classification of testing data
predicted_test = FCNN.classify_batch(X_test.to_numpy())

# Mean squared error
test_mse = mean_squared_error(y_test, predicted_test)
print("MSE (testing data) = ", test_mse)
# %%

# Model output and target output for training data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_train.x, X_train.y, predicted_train, cmap='Greens', s=0.2)
ax.scatter3D(X_train.x, X_train.y, y_train, cmap='Greens', s=0.2)

plt.xlabel("Input (x)")
plt.ylabel("Input (y)")
plt.title("Model output and target output for training data")
plt.legend(['Model Output', 'Target Output'])
plt.show()
# %%

# Target output vs model output for training data
plt.scatter(y_train, predicted_train)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Training)")
print("Correlation between target and predicted variable for training data")
print(np.corrcoef(y_train, predicted_train))
plt.show()


# %%
# Model output and target output for testing data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test.x, X_test.y, predicted_test, cmap='Greens', s=3)
ax.scatter3D(X_test.x, X_test.y, y_test, cmap='Greens', s=0.5)

plt.xlabel("Input (x)")
plt.ylabel("Input (y)")
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
print("Correlation between target and predicted variable for testing data")
print(np.corrcoef(y_test, predicted_test))


# %%

plt.bar(['Training', 'Validation', 'Testing'], [train_mse, val_mse,test_mse], color='maroon', width=0.4)
plt.title("Training and testing MSE")
plt.ylabel("MSE values")
plt.show()

#%%

fig = plt.figure(figsize=(10, 8))
fig.suptitle('Hidden Layer 1 (Validation Set)', y=0.02, fontsize=20)
for l in range(1,nodes_hidden_layer1+1):
    neuron_activation_value = []
    for i in range(X_val.shape[0]):
        neuron_activation_value.append(FCNN.classify_point(X_val.iloc[i].to_numpy())[1][1].gn[l])
        FCNN.clean_layers()

    ax = fig.add_subplot(2, 3, l, projection='3d', frame_on=True)
    ax.scatter3D(X_val.x, X_val.y, neuron_activation_value)

    plt.xlabel("Input (x)")
    plt.ylabel("Input (y)")
    plt.title("Neuron {}".format(l))

plt.show()

#%%
# fig = plt.figure(figsize=(10, 8))
# fig.suptitle('Hidden Layer 2 (Validation Set)', y=0.02, fontsize=20)
# for l in range(1,nodes_hidden_layer2+1):
#     neuron_activation_value = []
#     for i in range(X_val.shape[0]):
#         neuron_activation_value.append(FCNN.classify_point(X_val.iloc[i].to_numpy())[1][2].gn[l])
#         FCNN.clean_layers()

#     ax = fig.add_subplot(2, 3, l, projection='3d', frame_on=True)
#     ax.scatter3D(X_val.x, X_val.y, neuron_activation_value)

#     plt.xlabel("Input (x)")
#     plt.ylabel("Input (y)")
#     plt.title("Neuron {}".format(l))

# plt.show()

#%%
# ////////////////////////////////////////
# best arch: h1=6 h2=6
# stopping : e_difference <= 0.0001
# eta = 0.1

# ///////////////////////////////////////
# for 1 hidden layer having 15 nodes, the model could very well approximate the real function
# even for 1 hidden layer having only 5, the approximation is incredibly good