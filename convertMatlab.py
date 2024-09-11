import scipy.io
import pandas as pd
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt

data = loadmat("wbc.mat")
print(data)
df = pd.DataFrame()
arr = []

count = 1
for x in data:
    df = pd.concat([df,pd.DataFrame(data[x])])

print(df)
print(np.count_nonzero(np.array(df.iloc[:,-1:])))
plt.scatter(df.iloc[:,0],df.iloc[:,2], c=df.iloc[:,-1:], cmap='viridis')
plt.show()


