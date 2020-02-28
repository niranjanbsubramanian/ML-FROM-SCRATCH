import numpy as np
import matplotlib.pyplot as plt


def euc_dist(a, b):
    '''
    This method calculates the euclidean distance
    between two points
    '''
    return np.sqrt(np.sum((a-b)**2))


class KMeans():

    def __init__(self, K=3, max_iter=100, plot=False):
        '''
        Parameters
        -------------
        K: number of clusters
        max_iter: number of iterations
        plot: whether to plot the output cluster
        '''
        self.K = K
        self.max_iter = max_iter
        self.plot = plot        

    def fit(self, X):
        cluster = np.zeros(len(X))

        centroid = X[np.random.choice(len(X), self.K)]

        for i in range(self.max_iter):
        # assign points to cluster
            for idx, data in enumerate(X):
                # calculate distance between each point and the centroid
                dist = [euc_dist(c, data) for c in centroid]
                # assign the point to a cluster which has minimum distance
                cluster[idx] = np.argmin(dist)

            #recalculate cluster center points
            old_centroid = centroid
            # loop through the cluster and calculate the mean of all the points in the cluster and
            # make the means as the new centroids
            for i in range(self.K):
                pts_i = np.where(cluster == i)
                centroid[i] = np.mean(X[pts_i], axis=0)

        if self.plot:
            self.__plot_output(X, centroid, cluster)


    def __plot_output(self, X, centroid, cluster):
        '''
        This method will plot the clusters along with the centroids
        '''
        plt.scatter(X[:, 0], X[:, 1], s=50,c=cluster)
        plt.scatter(centroid[:,0], centroid[:,1], marker='*',c='red',s=200)
        plt.show()
