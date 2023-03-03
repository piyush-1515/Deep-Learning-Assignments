# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# %%
data_c1_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
data_c2_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class2.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
data_c3_train = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class3.txt", sep=" ", names=['x', 'y']).iloc[0:350, :]
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
    r"Group24\Group24\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
data_c2_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class2.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
data_c3_test = pd.read_csv(
    r"Group24\Group24\Classification\LS_Group24\Class3.txt", sep=" ", names=['x', 'y']).iloc[351:, :]
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

class Layer:

    num_nodes = int
    num_nodes_next = int
    num_nodes_prev = int
    previous_layer = None
    next_layer = None

    def __init__ (self, nodes):
        self.num_nodes = nodes

    def set_previous_layer(self, prev_layer):
        self.previous_layer = prev_layer
        self.num_nodes_prev = prev_layer.num_nodes

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer
        self.num_nodes_next = next_layer.num_nodes

#%%
class Model:
    num_layers = 0

    layers = [Layer]
    weights = []
    eta = int

    def __init__(self, eta):
        self.eta = eta

    def add_layer(self, layer):
        self.layers.append(layer)
        self.num_layers += 1
        if(self.num_layers >= 2):
            n1 = self.layers[self.num_layers-2].num_nodes
            n2 = layer.num_nodes
            self.weights.append(np.random.rand(n2,n1+1))

#%%
nodes_hidden_layer = 2
nodes_input_layer = 2
nodes_output_layer = 3

Layer_ip = Layer(nodes_input_layer)
Layer_hidden = Layer(nodes_hidden_layer)
Layer_output = Layer(nodes_output_layer)

Layer_ip.set_next_layer(Layer_hidden)
Layer_hidden.set_previous_layer(Layer_ip)
Layer_hidden.set_next_layer(Layer_output)
Layer_output.set_previous_layer(Layer_hidden)
