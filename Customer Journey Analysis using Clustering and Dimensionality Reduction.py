import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import plotly.graph_objects as go  # For interactive plots
warnings.filterwarnings('ignore')

#Loading Dataset
df = pd.read_csv('ecommerce_customer_data_large.csv')
df.head()

#Data Cleaning and Feature Engineering
df.drop_duplicates(inplace=True)
print(df.isnull().sum())
# Convert 'Purchase Date' to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

# Aggregate features per customer
features = df.groupby('Customer ID').agg({
    'Total Purchase Amount': 'sum',
    'Quantity': 'sum',
    'Returns': 'sum',
    'Product Price': 'mean',
    'Purchase Date': ['count', lambda x: (x.max() - x.min()).days] })

features.columns = ['Total_Spend', 'Total_Quantity', 'Total_Returns', 'Avg_Product_Price', 'Purchase_Frequency', 'Days_Between_First_Last']
features.reset_index(inplace=True)
features.head()

              #Normalization and PCA-Dimensionality Reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features.drop('Customer ID', axis=1))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
# Add PCA columns to features
features['PCA1'] = X_pca[:, 0]
features['PCA2'] = X_pca[:, 1]
features.head()

#K-Means Clustering and Cluster Visualization
kmeans = KMeans(n_clusters=3, random_state=42)
features['Cluster'] = kmeans.fit_predict(X_pca)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=features, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Customer Segments')
plt.show()

#Silhouette Score Evaluation
# Silhouette Score
silhouette_avg = silhouette_score(X_scaled, features['Cluster'])
print(f"\nSilhouette Score: {silhouette_avg}")
# PCA Feature Importance (Loadings)
print("\nPCA Component Loadings:")
print(pca.components_)
