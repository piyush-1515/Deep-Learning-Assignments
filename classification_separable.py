# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# %%
data_c1_train = pd.read_csv(
    r"Assignment1\Group19\Classification\LS_Group19\Class1.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
data_c2_train = pd.read_csv(
    r"Assignment1\Group19\Classification\LS_Group19\Class2.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
data_c3_train = pd.read_csv(
    r"Assignment1\Group19\Classification\LS_Group19\Class3.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
data_c1_train.insert(loc=0,
                     column='1',
                     value=[1 for i in range(350)])
data_c2_train.insert(loc=0,
                     column='1',
                     value=[1 for i in range(350)])
data_c3_train.insert(loc=0,
                     column='1',
                     value=[1 for i in range(350)])

# %%
data_c1_test = pd.read_csv(
    r"Assignment1\Group19\Classification\LS_Group19\Class1.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
data_c2_test = pd.read_csv(
    r"Assignment1\Group19\Classification\LS_Group19\Class2.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
data_c3_test = pd.read_csv(
    r"Assignment1\Group19\Classification\LS_Group19\Class3.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
data_c1_test.insert(loc=0,
                    column='1',
                    value=[1 for i in range(149)])
data_c2_test.insert(loc=0,
                    column='1',
                    value=[1 for i in range(149)])
data_c3_test.insert(loc=0,
                    column='1',
                    value=[1 for i in range(149)])

# %%

w12 = np.array([1.0, 1.0, 1.0])
w13 = np.array([1.0, 1.0, 1.0])
w23 = np.array([1.6,2.3,0.9])
inst_errors = []
avg_error_12 = []
avg_error_13 = []
avg_error_23 = []

def activation_function(a):
    f = 1 + np.exp(-a)
    return 1/f

def delta_w(eta, w, y, s, x):
    delta = eta * (y-s)*s*(1-s) * x
    return w + delta

def neuron(eta, w, x, y):
    a = np.dot(w, x)
    s = activation_function(a)
    E = ((y-s)**2)/2 # instantaneous error
    inst_errors.append(E)
    w = delta_w(eta, w, y, s, x)
    return w

# %%

# training class 1-2 perceptron
for epoch in range(1,21):
    eta = 1/epoch
    for i in range(350):
        w12 = neuron(eta, w12, data_c1_train.iloc[i], 0)
        w12 = neuron(eta, w12, data_c2_train.iloc[i], 1)
    avg_error_12.append(np.mean(inst_errors))
    inst_errors = []
    # if(epoch != 1 and (avg_error_12[epoch-1]-avg_error_12[epoch-2]) <0.001 ): break 

# %%
# training class 1-3 perceptron
for epoch in range(1,21):
    eta = 1/epoch
    for i in range(350):
        w13 = neuron(eta, w13, data_c1_train.iloc[i], 0)
        w13 = neuron(eta, w13, data_c3_train.iloc[i], 1)
    avg_error_13.append(np.mean(inst_errors))
    inst_errors = []
    # if(epoch != 1 and (avg_error_13[epoch-1]-avg_error_13[epoch-2]) <0.001 ): break 

# %%
# training class 2-3 perceptron
for epoch in range(1,21):
    eta = 1/epoch
    for i in range(350):
        w23 = neuron(eta, w23, data_c2_train.iloc[i], 0)
        w23 = neuron(eta, w23, data_c3_train.iloc[i], 1)
    avg_error_23.append(np.mean(inst_errors))
    inst_errors = []
    # if(epoch != 1 and (avg_error_23[epoch-1]-avg_error_23[epoch-2]) <0.00001 ): break 


#%%

plt.bar([i for i in range(1,21)], avg_error_12)
plt.title("Classifier between class 1 and class 2")
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.show()

plt.bar([i for i in range(1,21)], avg_error_13)
plt.title("Classifier between class 1 and class 3")
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.show()

plt.bar([i for i in range(1,21)], avg_error_23)
plt.title("Classifier between class 2 and class 3")
plt.xlabel("Epoch")
plt.ylabel("Average Error")
plt.show()

#%%
plt.scatter(data_c1_train.x, data_c1_train.y)
plt.scatter(data_c2_train.x, data_c2_train.y)
plt.scatter(data_c3_train.x, data_c3_train.y)
plt.legend(['Class 1', 'Class 2', 'Class 3'])
plt.axis('off')
plt.show()

#%%
# Creating one dataframe for testing data
classes = []
for i in range(3):
    for j in range(149):
        classes.append(i+1)
test_data = pd.concat([data_c1_test,data_c2_test,data_c3_test])
test_data['class'] = classes

#%%
# CLassifying the testing data

predicted_class = []

for i in range(447):
    freqs = []

    # binary clasification with 1-2 classifier
    a = np.dot(w12,test_data.iloc[i,:3])
    if (a<0) : freqs.append(1)
    else : freqs.append(2)
    
    # binary clasification with 1-3 classifier
    a = np.dot(w13,test_data.iloc[i,:3])
    if (a<0) : freqs.append(1)
    else : freqs.append(3)

    # binary clasification with 1-3 classifier
    a = np.dot(w23,test_data.iloc[i,:3])
    if (a<0) : freqs.append(2)
    else : freqs.append(3)

    predicted_class.append(np.bincount(freqs).argmax())

predicted_class = np.array(predicted_class)

#%%
from sklearn.metrics import accuracy_score
print("Accuracy = ",accuracy_score(test_data["class"], predicted_class))

from sklearn.metrics import confusion_matrix
print("Confusion Matrix = \n",confusion_matrix(test_data['class'], predicted_class))

#%%

# creating the mesh to achieve the decision boundaries

xmin = min(data_c1_train.min()[0+1], data_c2_train.min()[0+1], data_c3_train.min()[0+1])
ymin = min(data_c1_train.min()[1+1], data_c2_train.min()[1+1], data_c3_train.min()[1+1])

xmax = max(data_c1_train.max()[0+1], data_c2_train.max()[0+1], data_c3_train.max()[0+1])
ymax = max(data_c1_train.max()[1+1], data_c2_train.max()[1+1], data_c3_train.max()[1+1])

xx = np.linspace(xmin,xmax,50)
yy = np.linspace(ymin,ymax,50)

#%%
# Creating the decision boundary
predicted_mesh = pd.DataFrame(columns = ['x', 'y', 'pred'])

for i in xx:
    for j in yy:
        cord = np.array([1,i,j])
        
        freqs = []

        # binary clasification with 1-2 classifier
        a = np.dot(w12,cord)
        if (a<0) : freqs.append(1)
        else : freqs.append(2)
        
        # binary clasification with 1-3 classifier
        a = np.dot(w13,cord)
        if (a<0) : freqs.append(1)
        else : freqs.append(3)

        # binary clasification with 1-3 classifier
        a = np.dot(w23,cord)
        if (a<0) : freqs.append(2)
        else : freqs.append(3)

        # predicted_class.append(np.bincount(freqs).argmax())
        predicted_mesh = predicted_mesh.append({'x':cord[1], 'y':cord[2], 'pred':np.bincount(freqs).argmax()}, ignore_index=True)

#%%
training_data = pd.concat([data_c1_train,data_c2_train,data_c3_train])
classes = []
for i in range(3):
    for j in range(350):
        classes.append(i+1)
training_data['class'] = classes

#%%

fig, ax = plt.subplots()

scatter1 = ax.scatter(predicted_mesh['x'], predicted_mesh['y'], c=predicted_mesh['pred'],alpha=0.2)
scatter2 = ax.scatter(training_data['x'], training_data['y'], c=training_data['class'],alpha=0.8)
# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter1.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Decision boundaries and Training Data")
plt.show()