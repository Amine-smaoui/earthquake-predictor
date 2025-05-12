import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function for clustering
def cluster_earthquakes(combined, cluster_features):
    train_df = combined[combined['mag'].notnull()].copy()

    X_cluster = train_df[cluster_features].dropna()

    X_scaled = StandardScaler().fit_transform(X_cluster)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=2025)
    clusters = kmeans.fit_predict(X_scaled)

    X_cluster_labeled = X_cluster.copy()
    X_cluster_labeled['cluster'] = clusters

    # Add the cluster labels back to the original train_df (for further analysis)
    cluster_series = pd.Series(clusters, index=X_cluster.index)
    train_df['cluster'] = cluster_series

    return train_df, kmeans, X_cluster_labeled, clusters
