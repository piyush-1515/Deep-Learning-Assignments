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
data = pd.DataFrame(columns=['x', 'y'])

with open(r"Group24\Group24\Classification\NLS_Group24.txt") as f:
# with open(r"Group19\Group19\Classification\NLS_Group19.txt") as f:
# with open(r"Group18\Group18\Classification\NLS_Group18.txt") as f:
    cnt = 0
    while True:
        line = f.readline()
        if not line:
            break
        if(line[0] == 'F'):
            continue
        cords = line.split(' ')
        cords[0] = float(cords[0])
        cords[1] = float(cords[1])
        data = data.append({'x': cords[0], 'y': cords[1]}, ignore_index=True)

c1_data = data.iloc[:300, :]
c2_data = data.iloc[300:800, :]
c3_data = data.iloc[800:1800, :]

data_c1_train = c1_data.iloc[0:180, :]
data_c2_train = c2_data.iloc[0:300, :]
data_c3_train = c3_data.iloc[0:600, :]

# Creating one dataframe for training data
training_data = pd.concat([data_c1_train, data_c2_train, data_c3_train])
classes = []
for j in range(len(data_c1_train.x)):
    classes.append(1)
for j in range(len(data_c2_train.x)):
    classes.append(2)
for j in range(len(data_c3_train.x)):
    classes.append(3)
training_data['class'] = classes

data_c1_validation = c1_data.iloc[180:240, :]
data_c2_validation = c2_data.iloc[300:400, :]
data_c3_validation = c3_data.iloc[600:800, :]

# Creating one dataframe for validation data
validation_data = pd.concat([data_c1_validation, data_c2_validation, data_c3_validation])
classes = []
for j in range(len(data_c1_validation.x)):
    classes.append(1)
for j in range(len(data_c2_validation.x)):
    classes.append(2)
for j in range(len(data_c3_validation.x)):
    classes.append(3)
validation_data['class'] = classes

data_c1_test = c1_data.iloc[240:, :]
data_c2_test = c2_data.iloc[400:, :]
data_c3_test = c3_data.iloc[800:, :]
# Creating one dataframe for testing data
classes = []
for j in range(len(data_c1_test.x)):
    classes.append(1)
for j in range(len(data_c2_test.x)):
    classes.append(2)
for j in range(len(data_c3_test.x)):
    classes.append(3)
test_data = pd.concat([data_c1_test, data_c2_test, data_c3_test])
test_data['class'] = classes

# %%
# Making the model
epochs = 20
FCNN = Model(0.01)
FCNN.add_layer(2)
FCNN.add_layer(15)
FCNN.add_layer(30)
FCNN.add_layer(3)

# %%
# Fitting the model to the data
errors = []
while True:
    FCNN.fit(data_c1_train.to_numpy(), np.tile(np.array(
        [1, 0, 0]), data_c1_train.shape[0]).reshape(data_c1_train.shape[0], 3))
    FCNN.fit(data_c2_train.to_numpy(), np.tile(np.array(
        [0, 1, 0]), data_c2_train.shape[0]).reshape(data_c2_train.shape[0], 3))
    FCNN.fit(data_c3_train.to_numpy(), np.tile(np.array(
        [0, 0, 1]), data_c3_train.shape[0]).reshape(data_c3_train.shape[0], 3))
    # print("Average training error = ", FCNN.avg_training_error())
    errors.append(FCNN.avg_training_error())
    if(len(errors)>1 and abs(errors[len(errors)-1]-errors[len(errors)-2])<0.0001): break
    FCNN.total_error = []

# %%
# Plotting epoch vs training error
plt.plot([i for i in range(1, len(errors)+1)], errors)
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
plt.scatter(data_c1_train.x, data_c1_train.y, s=30, edgecolors='black')
plt.scatter(data_c2_train.x, data_c2_train.y, s=30, edgecolors='black')
plt.scatter(data_c3_train.x, data_c3_train.y, s=30, edgecolors='black')
plt.title('Dataset 2')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.grid()
plt.legend(['Class 1', 'Class 2', 'Class 3'])
# plt.axis('off')
# plt.show()

# %%
# Computing testing data classification accuacy
print("Accuracy = ", accuracy_score(test_data["class"], prediction))

print("Confusion Matrix = \n", confusion_matrix(
    test_data['class'], prediction))

# %%
# creating the mesh to achieve the decision boundaries
xmin = min(data_c1_train.min()[0], data_c2_train.min()[
        0], data_c3_train.min()[0])
ymin = min(data_c1_train.min()[1], data_c2_train.min()[
        1], data_c3_train.min()[1])

xmax = max(data_c1_train.max()[0], data_c2_train.max()[
        0], data_c3_train.max()[0])
ymax = max(data_c1_train.max()[1], data_c2_train.max()[
        1], data_c3_train.max()[1])

xx = np.linspace(xmin-2, xmax+2, 100)
yy = np.linspace(ymin-2, ymax+2, 100)

# %%
# Creating decision boundary between all classes
predicted_mesh = pd.DataFrame(columns=['x', 'y', 'pred'])
for i in xx:
    for j in yy:
        cord = np.array([i, j])
        predicted_mesh = predicted_mesh.append(
            {'x': cord[0], 'y': cord[1], 'pred': FCNN.classify_point(cord)[0]}, ignore_index=True)

# %%
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
# %%

