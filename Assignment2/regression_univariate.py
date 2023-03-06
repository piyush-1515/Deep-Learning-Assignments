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
nodes_hidden_layer = 6
FCNN.add_layer(1)
FCNN.add_layer(nodes_hidden_layer)
FCNN.add_layer(1)

errors = []
previous_validation_error = 100
inc_val_er_streak = 0
while True:
    FCNN.fit(X_train.to_numpy().reshape(len(X_train),1), y_train.to_numpy().reshape(len(y_train),1))
    errors.append(FCNN.avg_training_error())
    if((len(errors)>1 and abs(errors[len(errors)-1]-errors[len(errors)-2])<0.0001) or len(errors)>500): break
    
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
plt.plot([i for i in range(1,len(errors)+1)],errors)
plt.xlabel("Epoch")
plt.xlabel("Average Error")
plt.title("Epoch V/S Average error")
plt.show()

# %%
train_mse = 0
# Classification of training data
predicted = FCNN.classify_batch(X_train.to_numpy().reshape(len(X_train),1))

# Mean squared error
train_mse = mean_squared_error(y_train, predicted)
print("MSE (training data) = ", train_mse)

# %%

# Model output and target output for training data
plt.scatter(X_train, y_train)
plt.scatter(X_train, predicted)
plt.xlabel("Input (x)")
plt.ylabel("Predicted Output")
plt.title("Model output and target output for training data")
plt.legend(['Model Output', 'Target Output'])
plt.show()
# %%

# Target output vs model output for training data
plt.scatter(y_train, predicted)
plt.xlabel("Target output")
plt.ylabel("Predicted Output")
plt.title("Target output V/S model output (Training)")
plt.show()

# %%
test_mse = 0
# Classification of testing data
predicted = FCNN.classify_batch(X_test.to_numpy().reshape(len(X_test),1))

# Mean squared error
test_mse = mean_squared_error(y_test, predicted)
print("MSE (testing data) = ", test_mse)

# %%
# Model output and target output for testing data
plt.scatter(X_test, y_test)
plt.scatter(X_test, predicted)
plt.xlabel("Input (x)")
plt.ylabel("Predicted Output (x)")
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

# %%

plt.bar(['Training', 'Testing'], [train_mse, test_mse], color='maroon', width=0.4)
plt.title("Training and testing MSE")
plt.ylabel("MSE values")
plt.show()

#%%
