import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

print("Name: Anurag Gupta\nRegister Number: 22BIT0570")

# Load dataset
file_path = "Student_Mental_Stress_and_Coping_Mechanisms.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Define cluster values to test
cluster_values = [2, 4, 5, 7, 8,10]

for k in cluster_values:
    # Apply GMM with k clusters
    gmm = GaussianMixture(n_components=k, random_state=42)
    clusters = gmm.fit_predict(X_scaled)  # Assign cluster labels

    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score for k={k}: {silhouette_avg:.4f}")

    # Reduce Dimensions for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Visualize the Clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Cluster Label")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"GMM Clustering Visualization (k={k})")
    plt.show()

    # Show Number of Points per Cluster
    print(df.groupby(clusters).size())