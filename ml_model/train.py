import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os  

Current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(Current_dir, "data", "mall.csv")


# Load the dataset
def train_model():
    df = pd.read_csv(data_path)
    # Select relevant features (e.g., 'Annual Income' and 'Spending Score')
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Normalize the data if necessary
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
    joblib.dump(kmeans, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return kmeans, scaler

if __name__ == "__main__":
    train_model()