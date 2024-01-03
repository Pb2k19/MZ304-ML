import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data for clustering
X_moons, y_moons = make_moons(n_samples=400, noise=0.1, random_state=42)
X_circles, y_circles = make_circles(
    n_samples=400, factor=0.5, noise=0.1, random_state=42
)
X_classification, y_classification = make_classification(
    n_samples=400,
    n_features=2,
    n_classes=4,
    n_clusters_per_class=1,
    n_redundant=0,
    random_state=42,
)

# Scale the data
scaler = StandardScaler()
X_moons_scaled = scaler.fit_transform(X_moons)
X_circles_scaled = scaler.fit_transform(X_circles)
X_classification_scaled = scaler.fit_transform(X_classification)

# Apply clustering algorithms
kmeans = KMeans(n_clusters=2, random_state=42)
dbscan = DBSCAN(eps=0.3, min_samples=5)
hierarchical = AgglomerativeClustering(n_clusters=2)

labels_kmeans_moons = kmeans.fit_predict(X_moons_scaled)
labels_dbscan_moons = dbscan.fit_predict(X_moons_scaled)
labels_hierarchical_moons = hierarchical.fit_predict(X_moons_scaled)

labels_kmeans_circles = kmeans.fit_predict(X_circles_scaled)
labels_dbscan_circles = dbscan.fit_predict(X_circles_scaled)
labels_hierarchical_circles = hierarchical.fit_predict(X_circles_scaled)

labels_kmeans_classification = kmeans.fit_predict(X_classification_scaled)
labels_dbscan_classification = dbscan.fit_predict(X_classification_scaled)
labels_hierarchical_classification = hierarchical.fit_predict(X_classification_scaled)


# Evaluate clustering performance
def evaluate_clustering(X, labels):
    silhouette = silhouette_score(X, labels)
    ari = adjusted_rand_score(y_moons, labels)
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Adjusted Rand Index: {ari:.2f}")


# Evaluate for each dataset and algorithm
print("Evaluation for Moons Dataset:")
evaluate_clustering(X_moons_scaled, labels_kmeans_moons)
evaluate_clustering(X_moons_scaled, labels_dbscan_moons)
evaluate_clustering(X_moons_scaled, labels_hierarchical_moons)

print("\nEvaluation for Circles Dataset:")
evaluate_clustering(X_circles_scaled, labels_kmeans_circles)
evaluate_clustering(X_circles_scaled, labels_dbscan_circles)
evaluate_clustering(X_circles_scaled, labels_hierarchical_circles)

print("\nEvaluation for Classification Dataset:")
evaluate_clustering(X_classification_scaled, labels_kmeans_classification)
evaluate_clustering(X_classification_scaled, labels_dbscan_classification)
evaluate_clustering(X_classification_scaled, labels_hierarchical_classification)


# Visualization
def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolors="k", marker="o")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster")
    plt.show()


plot_clusters(
    X_moons_scaled, labels_kmeans_moons, "K-Means Clustering on Moons Dataset"
)
plot_clusters(X_moons_scaled, labels_dbscan_moons, "DBSCAN Clustering on Moons Dataset")
plot_clusters(
    X_moons_scaled,
    labels_hierarchical_moons,
    "Hierarchical Clustering on Moons Dataset",
)

plot_clusters(
    X_circles_scaled, labels_kmeans_circles, "K-Means Clustering on Circles Dataset"
)
plot_clusters(
    X_circles_scaled, labels_dbscan_circles, "DBSCAN Clustering on Circles Dataset"
)
plot_clusters(
    X_circles_scaled,
    labels_hierarchical_circles,
    "Hierarchical Clustering on Circles Dataset",
)

plot_clusters(
    X_classification_scaled,
    labels_kmeans_classification,
    "K-Means Clustering on Classification Dataset",
)
plot_clusters(
    X_classification_scaled,
    labels_dbscan_classification,
    "DBSCAN Clustering on Classification Dataset",
)
plot_clusters(
    X_classification_scaled,
    labels_hierarchical_classification,
    "Hierarchical Clustering on Classification Dataset",
)


# Dendrogram for Hierarchical Clustering
def plot_dendrogram(model, **kwargs):
    children = model.children_
    dist = np.arange(children.shape[0])
    no_of_observations = np.arange(2, children.shape[0] + 2)
    linkage_matrix = np.column_stack([children, dist, no_of_observations]).astype(float)
    dendrogram(linkage_matrix, **kwargs)


plt.title("Hierarchical Clustering Dendrogram for Moons Dataset")
plot_dendrogram(hierarchical, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
