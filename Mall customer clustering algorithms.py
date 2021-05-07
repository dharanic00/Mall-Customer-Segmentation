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
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
 
methods=['ward','single','complete','average','weighted','centroid','median']
 
plot_id=0
for method in methods:
    cl=linkage(reduced_data,method=method)
    
    for sw in ['dendrogram','clusters']:
        if sw=='dendrogram':
            plot_id+=1
            plt.subplot(7,2,plot_id)
            plt.title(method)
            fig,ax=plt.gcf(),plt.gca()
            dn=dendrogram(cl,truncate_mode='level',p=15)
            plt.tight_layout()
            fig.set_size_inches(10,15)
        else:
            plot_id+=1
            labels=fcluster(cl,2,criterion='maxclust')
            plt.subplot(7,2,plot_id)
            plt.title(method)
            plt.scatter(reduced_data.Dim1.values.tolist(),
                       reduced_data.Dim2.values.tolist(),
                       cmap=cmap,
                       c=labels)
plt.show() 
cl=linkage(reduced_data,method='ward')
fig,ax=plt.gcf(),plt.gca()
dn=dendrogram(cl,truncate_mode='level',p=15)
plt.tight_layout()
fig.set_size_inches(10,8)
plt.axhline(y=8,c='k')
plt.axhline(y=12,c='k')
plt.show()
from sklearn.cluster import DBSCAN
 
plot_id=0
for eps in np.arange(0.3,0.9,0.2):
    for min_samples in range(3,9):
        plot_id+=1
        cl=DBSCAN(eps=eps,min_samples=min_samples)
        result=cl.fit_predict(reduced_data)
        n_clusters=len([c for c in list(set(result)) if c!=-1])
        plt.subplot(6,4,plot_id)
        plt.scatter(reduced_data.Dim1.values.tolist(),
                   reduced_data.Dim2.values.tolist(),
                   cmap=cmap,
                   c=result)
        fig,ax=plt.gcf(),plt.gca()
        fig.set_size_inches(15,20)
        plt.title('eps: ' + str(eps)+', min_smp: ' + str(min_samples)+',\n# of clusters: ' + str(n_clusters))
        plt.tight_layout()
plt.show()
cl=DBSCAN(eps=0.7,min_samples=5)
result=cl.fit_predict(reduced_data)
n_clusters=len([c for c in list(set(result)) if c!=-1])
plt.scatter(reduced_data.Dim1.values.tolist(),
           reduced_data.Dim2.values.tolist(),
           cmap=cmap,
           c=result)
fig,ax=plt.gcf(),plt.gca()
fig.set_size_inches(5,5)
plt.title('eps :'+str(eps)+'min_smp :'+str(min_samples)+'n# of clusters :'+str(n_clusters))
plt.tight_layout()
#plt.savefig('img/dbscan_fav.png')
plt.show()
xs = data_pca2[:,0]
ys = data_pca2[:,1]
#zs = train_X.iloc[:,2]
plt.scatter(ys, xs)
#plt.scatter(ys, zs, c=labels)


plt.grid(False)
plt.title('Scatter Plot of Customers data')
plt.xlabel('PCA-01')
plt.ylabel('PCA-02')

plt.show()
k=4 
kmeans = KMeans(n_clusters=k, init = 'k-means++',random_state = 42) 
pipeline = make_pipeline(scaler, pca2, kmeans)
#pipeline = make_pipeline(kmeans)
# fit the model to the scaled dataset
model_fit = pipeline.fit(df2)
model_fit
labels = model_fit.predict(df2) //Assigning labels
Labels
# lets add the clusters to the dataset
train_X['Clusters'] = labels
# Number of data points for each feature in each cluster
train_X.groupby('Clusters').count()
xs = data_pca2[:,0]
ys = data_pca2[:,1]
#zs = train_X.iloc[:,2]
plt.scatter(ys, xs,c=labels)
#plt.scatter(ys, zs, c=labels)

plt.grid(False)
plt.title('Scatter Plot of Customers data')
plt.xlabel('PCA-01')
plt.ylabel('PCA-02')

plt.show()
centroids = model_fit[2].cluster_centers_
centroids
X = data_pca2
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
# Visualising the clusters & their Centriods
plt.figure(figsize=(15,7))
sns.scatterplot(X[labels == 0, 0], X[labels == 0, 1], color = 'grey', label = 'Cluster 1',s=50)
sns.scatterplot(X[labels == 1, 0], X[labels == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[labels == 2, 0], X[labels == 2, 1], color = 'yellow', label = 'Cluster 3',s=50)
sns.scatterplot(X[labels == 3, 0], X[labels == 3, 1], color = 'green', label = 'Cluster 4',s=50)

sns.scatterplot(centroids_x, centroids_y, color = 'red', 
                label = 'Centroids',s=300,marker='*')
plt.grid(False)
plt.title('Clusters of customers')
plt.xlabel('PCA-01')
plt.ylabel('PCA-02')
plt.legend()
plt.show()
model_fit[2].inertia_
plt.show()ks = range(1, 10)
wcss = []
samples = data_pca2

for i in ks:
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(samples)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10,5))
sns.lineplot(ks, wcss,marker='o',color='skyblue')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
def getInertia2(X,kmeans):
    ''' This function is analogous to getInertia, but with respect to the 2nd closest center, rather than closest one'''
    inertia2 = 0
    for J in range(len(X)):
        L = min(1,len(kmeans.cluster_centers_)-1) # this is just for the case where there is only 1 cluster at all
        dist_to_center = sorted([np.linalg.norm(X[J] - z)**2 for z in kmeans.cluster_centers_])[L]
        inertia2 = inertia2 + dist_to_center
    return inertia2 
wcss = []
inertias_2 = []
silhouette_avgs = []

ks = range(1, 10)
samples = data_pca2

for i in ks:
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(samples)
    wcss.append(kmeans.inertia_)
    inertias_2.append(getInertia2(samples,kmeans))
    if i>1:
        silhouette_avgs.append(silhouette_score(samples, kmeans.labels_))
silhouette_avgs
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
plt.title("wcss: sum square distances to closest cluster")
plt.plot(ks,wcss)
plt.xticks(ks)
plt.xlabel('number of clusters')
plt.grid()
    
plt.subplot(1,3,2)    
plt.title("Ratio: wcss VS. sum square distances to 2nd closest cluster")
plt.plot(ks,np.array(wcss)/np.array(inertias_2))
plt.xticks(ks)
plt.xlabel('number of clusters')
plt.grid()

plt.subplot(1,3,3)  
plt.title("Average Silhouette")
plt.plot(ks[1:], silhouette_avgs)
plt.xticks(ks)
plt.xlabel('number of clusters')
plt.grid()
plt.show()
df_new = test_X.copy()
# predict the labels
le.fit(df_new.Gender)

#update df2 with transformed values of gender
df_new.loc[:,'Gender'] = le.transform(df_new.Gender)

labels_test = model_fit.predict(df_new)
labels_test

plt.show()
test_X['Clusters'] = labels_test
# Number of data points for each feature in each cluster
test_X.groupby('Clusters').count()
query = (test_X['Clusters']==1)
test_X[query]
