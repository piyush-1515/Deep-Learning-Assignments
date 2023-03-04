#%%
import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#%%
a = [np.ndarray]

b = np.array([1,2,3])

#%%

a.append(b)
a.append(b)

#%%
a = np.array([[1,2],[3,4]])
# b = np.array([12,17])
b = [12,17]
np.dot(a,b)
#%%
def minus(x):
    return x-1
a = np.array([1,2,3,4,5])
minus(a)