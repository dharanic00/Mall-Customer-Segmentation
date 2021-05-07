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
data = pd.read_csv(r'C:\Users\charu\.spyder-py3\ML\Mall_Customers.csv')



data.rename(columns={'Annual Income (k$)':'AnnualIncome','Spending Score (1-100)':'SpendingScore'},inplace=True)
for i,col in enumerate(data.columns):
    print((i+1),'. columns is :',col)
data.drop('CustomerID',axis=1,inplace=True)
log_data=np.log(data)
good_data=log_data.drop([128,65,66,75,154])
good_data[:10]




