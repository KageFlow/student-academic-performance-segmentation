"""Cluster evaluation and model fitting helpers."""
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def evaluate_kmeans(X, k_values=range(2, 8), random_state=42):
    rows = []
    for k in k_values:
        if k >= len(X):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
        })
    return rows


def fit_cluster_models(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    hierarchical = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X)
    scores = {
        "kmeans_silhouette": float(silhouette_score(X, kmeans_labels)),
        "hierarchical_silhouette": float(silhouette_score(X, hierarchical_labels)),
    }
    return kmeans, kmeans_labels, hierarchical_labels, scores
