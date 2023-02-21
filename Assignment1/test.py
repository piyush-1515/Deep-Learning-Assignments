# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# %%
with open(r"Group24\Group24\Classification\NLS_Group24.txt") as f:
    cnt = 0
    while True:
        line = f.readline()
        if not line : break
        # if(line[0] == 'F') : 
        #     print("It was F word")
        #     continue
        print(line[0])
        # cords = line.split(' ')
        # cords[0] = float(cords[0])
        # cords[1] = float(cords[1])
        # data = data.append({'x':cords[0], 'y':cords[1]}, ignore_index=True)
