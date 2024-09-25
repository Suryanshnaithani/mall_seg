import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv(r'D:\my_ml_project\ml_model\mall.csv')

# Select relevant features (e.g., 'Annual Income' and 'Spending Score')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the data if necessary
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add the cluster labels to the dataframe
df['Cluster'] = clusters

# Visualize the clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()

#dump the model
import joblib
joblib.dump(kmeans, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')


