import numpy as np
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class OutlierDetection:

    def __init__(self):
        sns.set_theme()
        self.X, self.y = make_blobs(n_samples = 50, n_features = 2, centers = 4,cluster_std = 1.5, random_state = 4)
        print(self.X[0])
        output = self.main()
        print(self.X[output[1][0][1]], self.X[output[1][0][2]])
        print(  pow( pow(self.X[output[1][0][1]][0]-self.X[output[1][0][2]][0],2) +  pow(self.X[output[1][0][1]][1]-self.X[output[1][0][2]][1],2)  , .5) )

        self.printer()
        self.boxplot()

    def main(self):
        nn = NearestNeighbors(n_neighbors=3)
        nn.fit(self.X)
        knn = nn.kneighbors(self.X)  # returns 3 index neighbors including self
        return knn

    def printer(self):
        plt.scatter(self.X[:,0],self.X[:,1])

        plt.show()

    def boxplot(self):
        percent = np.quantile(self.X, [.25, .50, .75])
        print(percent)
        upperL = percent[2] + 1.5 * (percent[2] - percent[1])
        lowerL = percent[0] - 1.5 * (percent[1] - percent[0])
        print(upperL, lowerL)


outcome = OutlierDetection()
