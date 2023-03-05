import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

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
    num_layers = 0
    layers = []
    weights = []
    total_error = []
    eta = int

    def __init__(self, eta=1):
        self.eta = eta

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
            self.weights.append(np.random.rand(n2, n1+1))

    def update_weights(self, target):
        # updating weights between hidden and output layer
        wt_h_to_o = self.weights[1]
        delta_wjk = np.ones([len(wt_h_to_o), len(wt_h_to_o[0])], float)
        for k in range(len(wt_h_to_o)):
            for j in range(len(wt_h_to_o[0])):
                fa = self.layers[2].gn[k+1]
                hnj = self.layers[1].gn[j]
                ydiff = (target[k] - fa)
                delta_wjk[k][j] = self.eta * ydiff * fa * (1-fa) * hnj
        wt_h_to_o = wt_h_to_o + delta_wjk
        self.weights[1] = wt_h_to_o

        # updating weights between input and hidden layer
        wt_i_to_h = self.weights[0]
        delta_wij = np.ones([len(wt_i_to_h), len(wt_i_to_h[0])], float)
        for j in range(len(wt_i_to_h)):
            ydiff = 0
            gn = self.layers[1].gn
            for k in range(3):
                fa = self.layers[2].gn[k+1]
                ydiff += (target[k] - fa) * fa * (1-fa) * wt_h_to_o[k][j+1]
            for i in range(len(wt_i_to_h[0])):
                xni = self.layers[0].gn[i]
                delta_wij[j][i] = self.eta * ydiff * \
                    gn[j+1] * (1 - gn[j+1]) * xni
        wt_i_to_h = wt_i_to_h + delta_wij
        self.weights[0] = wt_i_to_h

    def update_wts(self,target):
        for iter in range(self.num_layers-2,-1,-1):

            wt_mat = self.weights[iter]
            delta_mat = np.ones([len(wt_mat), len(wt_mat[0])], float)

            if(iter == self.num_layers-2):
                for k in range(1,self.layers[iter+1].num_nodes+1):
                    fa = self.layers[iter+1].gn[k]
                    self.layers[iter+1].delta.append((fa - target[k-1]) * fa * (1 - fa))
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
                self.layers[wt_i+1].apply_act_func()

            inst_error = np.array(
                self.layers[self.num_layers-1].gn[1:]) - target[i]
            self.total_error.append(np.sum(np.square(inst_error))/2)
            # self.update_weights(target[i])
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
                self.layers[wt_i+1].apply_act_func()
            predictions.append(
                np.argmax(self.layers[self.num_layers-1].gn[1:])+1)
            self.clean_layers()
        return predictions
    
    def classify_point(self,x):
        xn = np.insert(x, 0, 1, axis=None)
        self.layers[0].gn = xn

        for wt_i in range(len(self.weights)):
            an = np.dot(self.weights[wt_i], self.layers[wt_i].gn)
            an = np.insert(an, 0, 1, axis=None)
            self.layers[wt_i+1].an = an
            self.layers[wt_i+1].apply_act_func()
        op = np.argmax(self.layers[self.num_layers-1].gn[1:])+1
        self.clean_layers()      
        return op

    def avg_training_error(self):
        return np.mean(self.total_error)

