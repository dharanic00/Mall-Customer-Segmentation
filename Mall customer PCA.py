# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:04:02 2021

@author: charu
"""
# demo: set working folder


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline   
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#for avoiding warnings
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv(r'C:\Users\charu\.spyder-py3\ML\Mall_Customers.csv')
df=df.drop('CustomerID',axis=1)
train_X, test_X = train_test_split(df, test_size=0.2, random_state=42)

print(len(train_X), "train +", len(test_X), "test")
df2 = train_X.copy()
le = LabelEncoder()
le.fit(df2.Gender)
le.classes_
df2.loc[:,'Gender'] = le.transform(df2.Gender)
df2.head(6)
scaler = StandardScaler()
scaler.fit(df2)
data_scaled = scaler.transform(df2)
data_scaled[0:3]
pca = PCA()

# fit PCA
pca.fit(data_scaled)
features = range(pca.n_components_)
features
data_pca = pca.transform(data_scaled)
data_pca.shape
pca.explained_variance_ratio_
plt.bar(features, pca.explained_variance_ratio_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
pca2 = PCA(n_components=2, svd_solver='full')

# fit PCA
pca2.fit(data_scaled)

# PCA transformed data
data_pca2 = pca2.transform(data_scaled)
data_pca2.shape






