import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import copy 
class Layer:

    num_nodes = int
    an = []
    gn = []
    delta = []

    def __init__(self, nodes):
        self.num_nodes = nodes

    def activation_function(self, a):
        f = 1 + np.exp(-a)
        return 1/f

    def apply_act_func(self):
        self.gn = self.activation_function(np.array(self.an))
        self.gn[0] = 1

class Model:
    mode = str
    num_layers = 0
    layers = []
    weights = []
    total_error = []
    eta = int
    validation_error = int
    validation_data_given = bool

    def __init__(self, eta=0.1, mode="class", validation_data=False):
        self.eta = eta
        self.mode = mode
        self.num_layers = 0
        self.layers = []
        self.weights = []
        self.total_error = []
        self.validation_data_given = validation_data

    def clean_layers(self):
        for l in self.layers:
            l.an = []
            l.gn = []
            l.delta = []

    def add_layer(self, num_nodes):
        new_layer = Layer(num_nodes)
        self.layers.append(new_layer)
        self.num_layers += 1
        if(self.num_layers >= 2):
            n1 = self.layers[self.num_layers-2].num_nodes
            n2 = new_layer.num_nodes
            self.weights.append(np.random.randn(n2, n1+1))

    def update_wts(self,target):
        for iter in range(len(self.weights)-1,-1,-1):

            wt_mat = self.weights[iter]
            delta_mat = np.ones([len(wt_mat), len(wt_mat[0])], float)

            if(iter == len(self.weights)-1):
                for k in range(1,self.layers[iter+1].num_nodes+1):
                    fa = self.layers[iter+1].gn[k]
                    if(self.mode == "class"): self.layers[iter+1].delta.append((target[k-1]-fa) * fa * (1 - fa))
                    else : self.layers[iter+1].delta.append((target[k-1]-fa) * 1)
            else :
                for l in range(1,self.layers[iter+1].num_nodes+1):
                    gnl = self.layers[iter+1].gn[l]
                    delta_nl = gnl * (1 - gnl)
                    sum_del = 0
                    for k in range(0,self.layers[iter+2].num_nodes):
                        wlk_mat = self.weights[iter+1]
                        sum_del += (self.layers[iter+2].delta[k] * wlk_mat[k][l])
                    delta_nl *= sum_del
                    self.layers[iter+1].delta.append(delta_nl)
                
            for k in range(len(wt_mat)):
                for j in range(len(wt_mat[0])):
                    hnj = self.layers[iter].gn[j]
                    delta = self.layers[iter+1].delta[k]
                    delta_mat[k][j] = self.eta * delta * hnj

            self.weights[iter] += delta_mat 
     
    def fit(self, x, target):
        for i in range(len(x)):
            xn = np.insert(x[i], 0, 1, axis=None)
            yn = target[i]
            self.layers[0].gn = xn

            for wt_i in range(len(self.weights)):
                an = np.dot(self.weights[wt_i], self.layers[wt_i].gn)
                an = np.insert(an, 0, 1, axis=None)
                self.layers[wt_i+1].an = an
                if(wt_i == self.num_layers-2 and self.mode=="regression") : self.layers[wt_i+1].gn = an
                else : self.layers[wt_i+1].apply_act_func()

            inst_error = np.array(
                self.layers[self.num_layers-1].gn[1:]) - target[i]
            self.total_error.append(np.sum(np.square(inst_error))/2)
            self.update_wts(target[i])
            self.clean_layers()

    def classify_batch(self, x):
        predictions = []
        for i in range(len(x)):
            xn = np.insert(x[i], 0, 1, axis=None)
            self.layers[0].gn = xn

            for wt_i in range(len(self.weights)):
                an = np.dot(self.weights[wt_i], self.layers[wt_i].gn)
                an = np.insert(an, 0, 1, axis=None)
                self.layers[wt_i+1].an = an
                if(wt_i == self.num_layers-2 and self.mode=="regression") : self.layers[wt_i+1].gn = an
                else : self.layers[wt_i+1].apply_act_func()
            if(self.mode == "class") : predictions.append(
                np.argmax(self.layers[self.num_layers-1].gn[1:])+1)
            else: predictions.append(self.layers[self.num_layers-1].gn[1])
            self.clean_layers()
        return predictions

    def classify_point(self, x):
        xn = np.insert(x, 0, 1, axis=None)
        self.layers[0].gn = xn

        for wt_i in range(len(self.weights)):
            an = np.dot(self.weights[wt_i], self.layers[wt_i].gn)
            an = np.insert(an, 0, 1, axis=None)
            self.layers[wt_i+1].an = an
            if(wt_i == self.num_layers-2 and self.mode=="regression") : self.layers[wt_i+1].gn = an
            else : self.layers[wt_i+1].apply_act_func()
        op = float
        if(self.mode == "class") : op = int(np.argmax(self.layers[self.num_layers-1].gn[1:])+1)
        else: op = self.layers[self.num_layers-1].gn[1]
        layers_data = copy.copy(self.layers)
        # self.clean_layers()
        return op,layers_data

    def avg_training_error(self):
        return np.mean(self.total_error)


