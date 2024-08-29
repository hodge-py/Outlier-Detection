import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd


class OutlierDetection:

    def __init__(self):
        self.outliers = []
        """
        self.X, self.y = make_blobs(n_samples=50, n_features=2, centers=3, cluster_std=2, random_state=2)
        self.X = self.X + 20
        """

        self.X, self.y = self.datasetSetup()

        output, dist = self.main()

        arr = self.generateArr(dist)

        # print(arr)
        self.boxplot(arr)
        self.printer(arr)

    def datasetSetup(self):
        df = pd.read_csv('Iris_with_outliers.csv')
        df = df.dropna()
        X = df.iloc[:, 2:4].values
        y = df.iloc[:, 6].values
        return X, y

    def main(self):
        nn = NearestNeighbors(n_neighbors=10)
        nn.fit(self.X, self.y)
        dist, knn = nn.kneighbors(self.X)  # returns 3 index neighbors including self
        return knn, dist

    def printer(self, dist):
        self.outliers = np.array(self.outliers)
        plt.subplot(1, 2, 1)
        plt.scatter(self.X[:, 0], self.X[:, 1])
        plt.scatter(self.outliers[:, 0], self.outliers[:, 1])
        plt.subplot(1, 2, 2)
        #plt.hist(self.X[:,0],bins=15)
        plt.boxplot(dist, vert=True)
        plt.show()

    def boxplot(self, dist):
        percent = np.quantile(dist, [.25, .50, .75])

        upperL = percent[2] + 1.5 * (percent[2] - percent[0])
        lowerL = percent[0] - 1.5 * (percent[2] - percent[0])

        for x in range(len(dist)):
            if dist[x] > upperL:
                self.outliers += [self.X[x]]
            elif dist[x] < lowerL:
                self.outliers += [self.X[x]]

    def generateArr(self, oriDist):
        arr = []
        for x in range(len(oriDist)):  # finds the distance away from that point (index 0)
            total = 0
            for y in range(len(oriDist[x])):
                # arr += [dist[x][0]+(dist[x][1]+dist[x][2])/2]
                total += oriDist[x][y]

            arr += [total]

        return arr


outcome = OutlierDetection()
