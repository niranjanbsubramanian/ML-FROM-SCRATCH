from sklearn.datasets import make_blobs
from kmeans import KMeans

X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.0)

km = KMeans(K=3, plot=True)
km.fit(X)
