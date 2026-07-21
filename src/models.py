import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import os

os.makedirs("../models", exist_ok=True)

def find_optimal_k(X_scaled, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"k={k} | Inercia: {km.inertia_:.1f} | Silhouette: {silhouettes[-1]:.4f}")
    return list(k_range), inertias, silhouettes

def train_kmeans(X_scaled, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    joblib.dump(model, "../models/kmeans.pkl")
    return model, labels

def train_dbscan(X_scaled, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} puntos ruido")
    return model, labels

def train_hierarchical(X_scaled, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    return model, labels

def evaluate_clustering(X_scaled, labels, name):
    if len(set(labels)) < 2:
        print(f"{name}: no se puede evaluar (menos de 2 clusters)")
        return {}
    metrics = {
        "Model": name,
        "N_Clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "Silhouette": silhouette_score(X_scaled, labels),
        "Davies_Bouldin": davies_bouldin_score(X_scaled, labels),
        "Calinski_Harabasz": calinski_harabasz_score(X_scaled, labels)
    }
    print(f"{name:20s} | Silhouette: {metrics['Silhouette']:.4f} | "
          f"DB: {metrics['Davies_Bouldin']:.4f} | CH: {metrics['Calinski_Harabasz']:.1f}")
    return metrics