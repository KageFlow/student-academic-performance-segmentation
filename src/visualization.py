"""Reusable visualizations for student clusters."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def plot_pca_clusters(X_scaled, labels):
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="viridis", alpha=0.8)
    ax.set(title="Student Clusters (PCA 2D)", xlabel="Principal Component 1", ylabel="Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    return fig, pca, coordinates


def plot_cluster_profiles(data, labels):
    frame = data.copy()
    frame["cluster"] = labels
    profile = frame.groupby("cluster")[[
        "study_hours", "attendance_rate", "assignments_submitted", "exam_score"
    ]].mean()
    ax = profile.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Average Student Features by Cluster")
    ax.set_ylabel("Feature average (original units)")
    ax.set_xlabel("Cluster")
    ax.tick_params(axis="x", rotation=0)
    fig = ax.get_figure()
    fig.tight_layout()
    return fig, profile
