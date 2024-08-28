import numpy as np
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import math


class OutlierDetection:

    def __init__(self):
        sns.set_theme()
        self.outliers = []
        self.X, self.y = make_blobs(n_samples = 50, n_features = 2, centers = 3,cluster_std = 2, random_state = 2)
        output, dist = self.main()

        arr = self.generateArr(dist)

        print(arr)
        self.boxplot(arr)
        self.printer(arr)

    def main(self):
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(self.X,self.y)
        dist, knn = nn.kneighbors(self.X)  # returns 3 index neighbors including self
        return knn, dist

    def printer(self,dist):
        self.outliers = np.array(self.outliers)
        plt.scatter(self.X[:,0],self.X[:,1])
        plt.scatter(self.outliers[:,0],self.outliers[:,1])
        #plt.hist(self.X[:,0],bins=10)
        #plt.boxplot(dist,vert=False)
        plt.show()

    def boxplot(self, dist):
        percent = np.quantile(dist, [.25, .50, .75])

        upperL = percent[2] + 1.5 * (percent[2] - percent[1])
        lowerL = percent[0] - 1.5 * (percent[1] - percent[0])
        print(upperL,lowerL)
        for x in range(len(dist)):
            if dist[x] > upperL:
                self.outliers += [self.X[x]]
            elif dist[x] < lowerL:
                self.outliers += [self.X[x]]

    def generateArr(self,oriDist):
        arr = []
        for x in range(len(oriDist)):  # finds the distance away from that point (index 0)
            total = 0
            for y in range(len(oriDist[x])):
                #arr += [dist[x][0]+(dist[x][1]+dist[x][2])/2]
                total += oriDist[x][y]

            arr += [total]

        return arr

outcome = OutlierDetection()
