# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:48:51 2020

@author: Admin
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('UpdatedDataset1.csv')
X = dataset.iloc[:, [12, 13]].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 0:34])
X[:, 0:34] = imputer.transform(X[:, 0:34])


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Drinking Water')
plt.ylabel('Euclidean distances')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Above Average Facilities')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Accepatable Facilities')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Poor Facilites')

plt.title('Water Facilities')
plt.xlabel('Drinking Water')
plt.ylabel('Safe Drinking Water')
plt.legend()
plt.show()
