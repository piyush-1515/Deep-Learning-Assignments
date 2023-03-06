# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import pandas as pd
import warnings
from fcnn import *
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
# Reading training data
data_c1_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
data_c2_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class2.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
data_c3_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class3.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]


# %%
# Reading testing data
data_c1_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
data_c2_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class2.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
data_c3_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class3.txt", sep=" ", names=['x', 'y']).iloc[351:, :]

# %%


# %%
# Model

# %%
# Making the model
nodes_input_layer = 2
nodes_hidden_layer = 50
nodes_output_layer = 3
epochs = 20
FCNN = Model()
FCNN.add_layer(nodes_input_layer)
FCNN.add_layer(10)
# FCNN.add_layer(10)
# FCNN.add_layer(2)
FCNN.add_layer(nodes_output_layer)

# %%
# Fitting the model to the data
errors = []
for epoch in range(1, epochs+1):
    FCNN.eta = 1/epoch
    FCNN.fit(data_c1_train.to_numpy(), np.tile(np.array(
        [1, 0, 0]), data_c1_train.shape[0]).reshape(data_c1_train.shape[0], 3))
    FCNN.fit(data_c2_train.to_numpy(), np.tile(np.array(
        [0, 1, 0]), data_c2_train.shape[0]).reshape(data_c2_train.shape[0], 3))
    FCNN.fit(data_c3_train.to_numpy(), np.tile(np.array(
        [0, 0, 1]), data_c3_train.shape[0]).reshape(data_c3_train.shape[0], 3))
    # print("Average training error = ", FCNN.avg_training_error())
    errors.append(FCNN.avg_training_error())

# %%
# Plotting epoch vs training error
plt.plot([i for i in range(1, epochs+1)], errors)
plt.title("Training error vs epoch")
plt.xlabel("Epochs")
plt.ylabel("Average Error")
plt.grid()
plt.show()

# %%
# Classifying the testing data
prediction = FCNN.classify_batch(data_c1_test.to_numpy())
prediction = prediction + FCNN.classify_batch(data_c2_test.to_numpy())
prediction = prediction + FCNN.classify_batch(data_c3_test.to_numpy())

# %%
# Plotting the training data
plt.scatter(data_c1_train.x, data_c1_train.y, s=10)
plt.scatter(data_c2_train.x, data_c2_train.y, s=10)
plt.scatter(data_c3_train.x, data_c3_train.y, s=10)
plt.title('Dataset 1')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.grid()
plt.legend(['Class 1', 'Class 2', 'Class 3'])
# plt.axis('off')
plt.show()

# %%
# Creating one dataframe for testing data
classes = []
for i in range(3):
    for j in range(149):
        classes.append(i+1)
test_data = pd.concat([data_c1_test, data_c2_test, data_c3_test])
test_data['class'] = classes

# %%
# Computing testing data classification accuacy
print("Accuracy = ", accuracy_score(test_data["class"], prediction))

print("Confusion Matrix = \n", confusion_matrix(
    test_data['class'], prediction))

# %%
# creating the mesh to achieve the decision boundaries
xmin = min(data_c1_train.min()[0], data_c2_train.min()[0], data_c3_train.min()[0])
ymin = min(data_c1_train.min()[1], data_c2_train.min()[1], data_c3_train.min()[1])

xmax = max(data_c1_train.max()[0], data_c2_train.max()[0], data_c3_train.max()[0])
ymax = max(data_c1_train.max()[1], data_c2_train.max()[1], data_c3_train.max()[1])

xx = np.linspace(xmin-2, xmax+2, 100)
yy = np.linspace(ymin-2, ymax+2, 100)

#%%
# Combining training data
training_data = pd.concat([data_c1_train, data_c2_train, data_c3_train])
classes = []
for i in range(3):
    for j in range(350):
        classes.append(i+1)
training_data['class'] = classes

#%%
# Creating decision boundary between all classes
predicted_mesh = pd.DataFrame(columns=['x', 'y', 'pred'])
for i in xx:
    for j in yy:
        cord = np.array([i, j])
        predicted_mesh = predicted_mesh.append(
                {'x': cord[0], 'y': cord[1], 'pred': FCNN.classify_point(cord)}, ignore_index=True)

#%%
# Superimposing the decision boundary onto the training data
fig, ax = plt.subplots()

scatter1 = ax.scatter(
    predicted_mesh['x'], predicted_mesh['y'], c=predicted_mesh['pred'], alpha=0.2, s=10)
scatter2 = ax.scatter(
    training_data['x'], training_data['y'], c=training_data['class'], alpha=1, s=10, edgecolors='black')
# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter1.legend_elements(),
                    loc="lower left", title="Classes")
plt.grid()
ax.add_artist(legend1)

plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("Decision Regions and Training Data")
plt.show()
#%%
