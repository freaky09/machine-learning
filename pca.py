import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


dataset=load_breast_cancer()
df=pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
print(df.head())
scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
print(scaled_data)
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=dataset['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

