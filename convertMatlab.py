import scipy.io
import pandas as pd
import numpy as np
from mat4py import loadmat
import matplotlib.pyplot as plt

data = loadmat("t4_8K.mat")
print(data)
df = pd.DataFrame()
arr = []

count = 1
for x in data:
    df = pd.concat([df,pd.DataFrame(data[x])],axis=1)

print(df)

plt.scatter(df.iloc[:,0],df.iloc[:,1], c=df.iloc[:,-1:])
plt.show()


