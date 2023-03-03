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


#%%
class Layer:

    num_nodes = int
    an = []
    gn = []
    def __init__ (self, nodes):
        self.num_nodes = nodes

    def activation_function(self,a):
        f = 1 + np.exp(-a)
        return 1/f
    
    def apply_act_func(self):
        for ani in self.an:
            self.gn.append(self.activation_function(ani))
#%%
class Model:
    num_layers = 0

    layers = []\
    weights = []
    eta = int

    def __init__(self, eta=1):
        self.eta = eta

    def add_layer(self, num_nodes):
        new_layer = Layer(num_nodes)
        self.layers.append(new_layer)
        self.num_layers += 1
        if(self.num_layers >= 2):
            n1 = self.layers[self.num_layers-2].num_nodes
            n2 = new_layer.num_nodes
            self.weights.append(np.random.rand(n2,n1+1))

    def update_weights(self,target):
        # updating weights between hidden and output layer
        wt_h_to_o = self.weights[1]
        delta_wjk = np.ones([len(wt_h_to_o),len(wt_h_to_o[0])], float)
        for k in range(len(wt_h_to_o)):
            for j in range(len(wt_h_to_o[0])):
                fa = self.layers[2].gn[k+1]
                hnj = self.layers[1].hn[j]
                delta_wjk[j][k] = self.eta * (target - fa) * fa * (1-fa) * hnj
        wt_h_to_o = wt_h_to_o + delta_wjk

        # updating weights between input and hidden layer
        wt_i_to_h = self.weights[0]
        delta_wij = np.ones([len(wt_i_to_h),len(wt_i_to_h[0])], float)
        for j in range(len(wt_i_to_h)):
            for i in range(len(wt_i_to_h[0])):
                ydiff = 0
                fa = self.layers[2].gn[k+1]
                for k in range(3):
                    ydiff += (target - self.layers[2].gn[k+1]) * fa * (1-fa) * wt_h_to_o[j][k+1]
                
                ga = self.layers[1].gn[k]
                hnj = self.layers[1].hn[j]
                delta_wjk[j][k] = self.eta * (target - fa) * fa * (1-fa) * hnj
        wt_h_to_o = wt_h_to_o + delta_wjk

    def fit(self, x, target):
        a = 0
        for i in range(len(x)):
            xn = np.insert(x[i], 0, 1, axis=None)
            yn = target[i]
            self.layers[0].gn = x[i]

            for wt_i in range(len(self.weights)):
                # prev_neuron_out = np.insert(self.layers[wt_i].gn, 0, 1, axis=None)
                an = np.dot(self.weights[wt_i],self.layers[wt_i].gn)
                # an = np.dot(self.weights[wt_i],prev_neuron_out)
                an = np.insert(an, 0, 1, axis=None)
                self.layers[wt_i+1].an = an
                self.layers[wt_i+1].apply_act_func()
            
            self.update_weights(target)

#%%
nodes_hidden_layer = 2
nodes_input_layer = 2
nodes_output_layer = 3

FCNN = Model()
FCNN.add_layer(nodes_input_layer)
FCNN.add_layer(nodes_hidden_layer)
FCNN.add_layer(nodes_output_layer)
