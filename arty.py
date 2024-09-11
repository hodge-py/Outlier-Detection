import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


arr = np.random.multivariate_normal([50,45], [[20,0],[0,17]], 1000)

print(arr)

plt.scatter(arr[:,0],arr[:,1])
plt.show()

