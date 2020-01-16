import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import KNearestNeighbors

iris = load_iris()
data = iris.data    
target = iris.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=5656)

clf = KNearestNeighbors(K=3)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print('Accuracy:', accuracy_score(y_test, predictions))