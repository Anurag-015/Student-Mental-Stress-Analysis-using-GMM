import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Load Data
file_path = "Student_Mental_Stress_and_Coping_Mechanisms.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Handle Missing Values (Mean Imputation)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Normalize the Data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# Define GMM Variants
covariance_types = ["full", "tied", "diag", "spherical"]
bic_scores = {}

for cov_type in covariance_types:
    # Determine Optimal Number of Clusters using BIC
    bic_values = []
    cluster_range = range(2, 10)
    
    for k in cluster_range:
        gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42)
        gmm.fit(X_scaled)
        bic_values.append(gmm.bic(X_scaled))
    
    # Select the best k based on the minimum BIC score
    optimal_k = cluster_range[np.argmin(bic_values)]
    
    # Apply GMM with Optimal k
    gmm = GaussianMixture(n_components=optimal_k, covariance_type=cov_type, random_state=42)
    clusters = gmm.fit_predict(X_scaled)
    
    bic_scores[cov_type] = min(bic_values)
    
    print(f"Covariance Type: {cov_type}, Optimal k: {optimal_k}, BIC: {bic_scores[cov_type]}")
    
    # Reduce Dimensions for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualize the Clustering
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(label="Cluster Label")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"GMM ({cov_type}) Clustering Visualization (k={optimal_k})")
    plt.show()

# Print Summary of Best Variant
best_variant = min(bic_scores, key=bic_scores.get)
print(f"Best GMM Variant: {best_variant} with BIC: {bic_scores[best_variant]}")
