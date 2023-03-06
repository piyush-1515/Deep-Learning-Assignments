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

X_train, X_test, y_train, y_test = train_test_split(
    data, target_output, test_size=0.30, random_state=42)

# %%

FCNN = Model(0.1,"regression")
FCNN.add_layer(2)
FCNN.add_layer(5)
FCNN.add_layer(5)
FCNN.add_layer(5)
FCNN.add_layer(1)

errors = []
while True:
    FCNN.fit(X_train.to_numpy(), y_train.to_numpy().reshape(len(y_train),1))
    errors.append(FCNN.avg_training_error())
    if(len(errors)>1 and abs(errors[len(errors)-1]-errors[len(errors)-2])<0.0001): break

# %%
plt.plot(errors)
plt.xlabel("Epoch")
plt.xlabel("Average Error")
plt.title("Epoch V/S Average error")
plt.show()

# %%
# Classification of training data
train_mse = 0
predicted = FCNN.classify_batch(X_train.to_numpy())

# Mean squared error
train_mse = mean_squared_error(y_train, predicted)
print("MSE (training data) = ", train_mse)

# %%

# Model output and target output for training data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_train.x, X_train.y, predicted, cmap='Greens')
ax.scatter3D(X_train.x, X_train.y, y_train, cmap='Greens', alpha=0.1)

plt.xlabel("Input (x)")
plt.ylabel("Input (y)")
plt.title("Model output and target output for training data")
plt.legend(['Model Output', 'Target Output'])
plt.show()
# %%

# Target output vs model output for training data
plt.scatter(y_train, predicted)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Training)")
print("Correlation between target and predicted variable for training data")
print(np.corrcoef(y_train, predicted))
plt.show()

# %%
test_mse = 0
# Classification of testing data
predicted = FCNN.classify_batch(X_test.to_numpy())

# Mean squared error
test_mse = mean_squared_error(y_test, predicted)
print("MSE (testing data) = ", test_mse)

# %%
# Model output and target output for testing data
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X_test.x, X_test.y, predicted, cmap='Greens')
ax.scatter3D(X_test.x, X_test.y, y_test, cmap='Greens', alpha=0.1)

plt.xlabel("Input (x)")
plt.ylabel("Input (y)")
plt.title("Model output and target output for testing data")
plt.legend(['Model Output', 'Target Output'])
plt.show()


# %%

# Target output vs model output for testing data
plt.scatter(y_test, predicted)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Testing)")
plt.show()
print("Correlation between target and predicted variable for testing data")
print(np.corrcoef(y_test, predicted))

# %%

plt.bar(['Training', 'Testing'], [train_mse, test_mse], color='maroon', width=0.4)
plt.title("Training and testing MSE")
plt.ylabel("MSE values")
plt.show()

#%%