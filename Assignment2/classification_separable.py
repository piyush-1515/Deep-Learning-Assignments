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
    r"Group24\Group24\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y']).iloc[0:300, :]
data_c2_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class2.txt", sep=" ", names=['x', 'y']).iloc[0:300, :]
data_c3_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class3.txt", sep=" ", names=['x', 'y']).iloc[0:300, :]

# %%
# Reading validation data
data_c1_validation = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y']).iloc[300:400, :]
data_c2_validation = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class2.txt", sep=" ", names=['x', 'y']).iloc[300:400, :]
data_c3_validation = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class3.txt", sep=" ", names=['x', 'y']).iloc[300:400, :]
# Creating one dataframe for validation data
classes = []
for i in range(3):
    for j in range(len(data_c1_validation)):
        classes.append(i+1)
validation_data = pd.concat([data_c1_validation, data_c2_validation, data_c3_validation])
validation_data['class'] = classes
# %%
# Reading testing data
data_c1_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y']).iloc[400:, :]
data_c2_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class2.txt", sep=" ", names=['x', 'y']).iloc[400:, :]
data_c3_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class3.txt", sep=" ", names=['x', 'y']).iloc[400:, :]

# %%
# Making the model
nodes_input_layer = 2
nodes_hidden_layer = 10
nodes_output_layer = 3
epochs = 20
FCNN = Model(0.1)
FCNN.add_layer(nodes_input_layer)
FCNN.add_layer(nodes_hidden_layer)
FCNN.add_layer(nodes_output_layer)

# %%
# Fitting the model to the data
errors = []
previous_validation_error = 100
inc_val_er_streak = 0
for epoch in range(1, epochs+1):
    # FCNN.eta = 1/epoch
    FCNN.fit(data_c1_train.to_numpy(), np.tile(np.array(
        [1, 0, 0]), data_c1_train.shape[0]).reshape(data_c1_train.shape[0], 3))
    FCNN.fit(data_c2_train.to_numpy(), np.tile(np.array(
        [0, 1, 0]), data_c2_train.shape[0]).reshape(data_c2_train.shape[0], 3))
    FCNN.fit(data_c3_train.to_numpy(), np.tile(np.array(
        [0, 0, 1]), data_c3_train.shape[0]).reshape(data_c3_train.shape[0], 3))
    errors.append(FCNN.avg_training_error())

    # Computing the validation error
    prediction = FCNN.classify_batch(data_c1_validation.to_numpy())
    prediction += FCNN.classify_batch(data_c2_validation.to_numpy())
    prediction += FCNN.classify_batch(data_c3_validation.to_numpy())
    current_validation_error = accuracy_score(validation_data["class"], prediction)
    if(previous_validation_error < current_validation_error): 
        inc_val_er_streak += 1
        if(inc_val_er_streak >= 3): break
    inc_val_er_streak = 0        
    previous_validation_error = current_validation_error

# %%
# Plotting epoch vs training error
plt.plot([i for i in range(len(errors))], errors)
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
    for j in range(len(data_c1_test)):
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
    for j in range(len(data_c1_train)):
        classes.append(i+1)
training_data['class'] = classes

#%%
# Creating decision boundary between all classes
predicted_mesh = pd.DataFrame(columns=['x', 'y', 'pred'])
for i in xx:
    for j in yy:
        cord = np.array([i, j])
        predicted_mesh = predicted_mesh.append(
                {'x': cord[0], 'y': cord[1], 'pred': FCNN.classify_point(cord)[0]}, ignore_index=True)
        FCNN.clean_layers()

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

# plotting outputs of each of the hidden layer neurons
for l in range(1,nodes_hidden_layer):
    neuron_activation_value = []
    for i in range(training_data.shape[0]):
        neuron_activation_value.append(FCNN.classify_point(training_data.iloc[i,0:2].to_numpy())[1][1].gn[l])
        FCNN.clean_layers()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(training_data.x, training_data.y, neuron_activation_value, c=training_data["class"])

    plt.xlabel("Input (x)")
    plt.ylabel("Input (y)")
    plt.title("Output of the hidden layer neurons")
    # plt.legend(['Model Output', 'Target Output'])
    # ax.legend(["Class-1", "Class-2", "Class-3"])
    plt.show()
    plt.close()

#%%
