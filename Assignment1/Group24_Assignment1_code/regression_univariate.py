# %%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# %%
data = pd.read_csv(r'Group24\Group24\Regression\UnivariateData\24.csv', names=['x', 'y'])
X, Y = data.x, data.y
# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)
X_train = pd.DataFrame(
    {'1': [1 for i in range(X_train.shape[0])], 'x': X_train})
X_test = pd.DataFrame({'1': [1 for i in range(X_test.shape[0])], 'x': X_test})
y_train = y_train.reset_index()
y_test = y_test.reset_index()
y_train = y_train.drop(['index'], axis=1)
y_test = y_test.drop(['index'], axis=1)
# %%

w = np.array([1.0, 1.0])
inst_errors = []
avg_error = []


def activation_function(a):
    return a


def delta_w(eta, w, y, s, x):
    delta = eta * (y-s) * x
    return w + delta


def neuron(eta, w, x, y):
    a = np.dot(w, x)
    s = activation_function(a)
    E = ((y-s)**2)/2  # instantaneous error
    inst_errors.append(E)
    w = delta_w(eta, w, y, s, x)
    return w

# %%


# Training the model
for epoch in range(1, 21):
    eta = 1/epoch
    for i in range(X_train.shape[0]):
        w = neuron(eta, w, X_train.iloc[i], y_train.iloc[i, 0])
    avg_error.append(np.mean(inst_errors))
    inst_errors = []

# %%
plt.plot(avg_error)
plt.xlabel("Epoch")
plt.xlabel("Average Error")
plt.title("Epoch V/S Average error")
plt.show()

# %%
train_mse = 0
# Classification of training data
predicted = []
for i in range(X_train.shape[0]):
    predicted.append(np.dot(X_train.iloc[i], w))

# Mean squared error
train_mse = mean_squared_error(y_train, predicted)
print("MSE (training data) = ", train_mse)

# %%

# Model output and target output for training data
plt.scatter(X_train.x, y_train.y)
plt.scatter(X_train.x, predicted)
plt.xlabel("Input (x)")
plt.ylabel("Predicted Output")
plt.title("Model output and target output for training data")
plt.legend(['Model Output', 'Target Output'])
plt.show()
# %%

# Target output vs model output for training data
plt.scatter(y_train.y, predicted)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Training)")
plt.show()

# %%
test_mse = 0
# Classification of testing data
predicted = []
for i in range(X_test.shape[0]):
    predicted.append(np.dot(X_test.iloc[i], w))

# Mean squared error
test_mse = mean_squared_error(y_test, predicted)
print("MSE (testing data) = ", test_mse)

# %%
# Model output and target output for testing data
plt.scatter(X_test.x, y_test.y)
plt.scatter(X_test.x, predicted)
plt.xlabel("Input (x)")
plt.ylabel("Predicted Output (x)")
plt.title("Model output and target output for testing data")
plt.legend(['Model Output', 'Target Output'])
plt.show()

# %%

# Target output vs model output for testing data
plt.scatter(y_test.y, predicted)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Testing)")
plt.show()

# %%

plt.bar(['Training', 'Testing'], [train_mse, test_mse], color='maroon', width=0.4)
plt.title("Training and testing MSE")
plt.ylabel("MSE values")
plt.show()
