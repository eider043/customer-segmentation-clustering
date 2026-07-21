"""
Segmentacion de Clientes — Clustering
K-Means, DBSCAN y Clustering Jerarquico

Autor: Eider
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data, feature_matrix
from preprocessing import scale_features
from models import (find_optimal_k, train_kmeans, train_dbscan,
                    train_hierarchical, evaluate_clustering)
from evaluate import (plot_elbow_silhouette, plot_clusters_2d,
                      plot_cluster_profiles, plot_dendrogram, plot_comparison)

def main():
    print("=" * 60)
    print("SEGMENTACION DE CLIENTES — CLUSTERING")
    print("K-Means | DBSCAN | Clustering Jerarquico")
    print("=" * 60)

    print("\n[1/5] Cargando datos...")
    df = load_data()
    X = feature_matrix(df)
    X_scaled, scaler = scale_features(X)

    print("\n[2/5] Encontrando K optimo...")
    k_range, inertias, silhouettes = find_optimal_k(X_scaled)
    best_k = plot_elbow_silhouette(k_range, inertias, silhouettes)

    print(f"\n[3/5] Entrenando modelos con K={best_k}...")
    km_model, km_labels = train_kmeans(X_scaled, n_clusters=best_k)
    db_model, db_labels = train_dbscan(X_scaled, eps=0.4, min_samples=5)
    hc_model, hc_labels = train_hierarchical(X_scaled, n_clusters=best_k)

    print("\n[4/5] Evaluando algoritmos...")
    metrics = [
        evaluate_clustering(X_scaled, km_labels, "K-Means"),
        evaluate_clustering(X_scaled, hc_labels, "Jerarquico"),
    ]
    if len(set(db_labels)) > 1:
        valid_mask = db_labels != -1
        metrics.append(evaluate_clustering(
            X_scaled[valid_mask], db_labels[valid_mask], "DBSCAN"
        ))

    metrics_df = pd.DataFrame([m for m in metrics if m])
    metrics_df.to_csv("../outputs/clustering_metrics.csv", index=False)

    print("\n[5/5] Generando visualizaciones...")
    df["Cluster_KMeans"] = km_labels
    df["Cluster_DBSCAN"] = db_labels
    df["Cluster_HC"] = hc_labels

    plot_clusters_2d(df, km_labels)
    plot_cluster_profiles(df, km_labels)
    plot_dendrogram(X_scaled)
    plot_comparison(metrics)

    df.to_csv("../data/customers_clustered.csv", index=False)
    print("\nProceso completado.")
    print("Para el dashboard ejecuta: streamlit run dashboard.py")

if __name__ == "__main__":
    main()