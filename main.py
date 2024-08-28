import numpy as np
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import math


class OutlierDetection:

    def __init__(self):
        sns.set_theme()
        self.X, self.y = make_blobs(n_samples = 50, n_features = 2, centers = 3,cluster_std = 2, random_state = 4)
        output, dist = self.main()
        print(dist)
        arr = []
        for x in range(len(dist)):
            arr += [dist[x][0]+dist[x][1]+dist[x][2]]
        print(arr)
        self.printer(arr)
        self.boxplot(arr)

    def main(self):
        nn = NearestNeighbors(n_neighbors=3)
        nn.fit(self.X,self.y)
        dist, knn = nn.kneighbors(self.X)  # returns 3 index neighbors including self
        return knn,dist

    def printer(self,dist):
        plt.scatter(self.X[:,0],self.X[:,1])
        #plt.boxplot(dist)
        plt.show()

    def boxplot(self, dist):
        percent = np.quantile(dist, [.25, .50, .75])
        #print(percent)
        upperL = percent[2] + 1.5 * (percent[2] - percent[1])
        lowerL = percent[0] - 1.5 * (percent[1] - percent[0])
        print(len(dist))
        for x in range(len(dist)):
            if dist[x] > upperL:
                print(self.X[x])
            elif dist[x] < lowerL:
                print(self.X[x])



outcome = OutlierDetection()
