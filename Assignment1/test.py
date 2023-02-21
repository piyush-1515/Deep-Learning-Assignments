# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# %%
data_c1_train = pd.read_csv(
    r"data\Classification\LS_Group24\Class1.txt", sep=" ", names=['x', 'y'])

data = pd.read_csv(
    r"data\Classification\NLS_Group24.txt", sep=" ", names=['x', 'y'])