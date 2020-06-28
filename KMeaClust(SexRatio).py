# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:37:59 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:16:38 2020

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 31 00:12:59 2020

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('UpdatedDataset1.csv')
X = dataset.iloc[:, [2, 3]].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 0:33])
X[:, 0:33] = imputer.transform(X[:, 0:33])




from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Best Sex Ratio')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Good Sex Ratio')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Bad Sex Ratio')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Change In Sex Ratio')
plt.xlabel('Sex Ratio')
plt.ylabel('Sex Ratio Under 6 Years')
plt.legend()
plt.show()
