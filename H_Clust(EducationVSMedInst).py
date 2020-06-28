# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:10:02 2020

@author: Admin
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('UpdatedDataset1.csv')
X = dataset.iloc[:, [30, 32]].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 0:34])
X[:, 0:34] = imputer.transform(X[:, 0:34])

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Educational Institutions')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Poor Facilities')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Accepatable Facilities')


plt.title('Table Of Institutions')
plt.xlabel('Educational Institutions')
plt.ylabel('Medical Institutions')
plt.legend()
plt.show()
