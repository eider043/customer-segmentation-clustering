import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import os

os.makedirs("../outputs", exist_ok=True)

CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

def plot_elbow_silhouette(k_range, inertias, silhouettes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, inertias, "bo-", lw=2, markersize=8)
    axes[0].set_title("Metodo del Codo — Inercia por K", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Numero de clusters (K)")
    axes[0].set_ylabel("Inercia (WCSS)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(k_range, silhouettes, "rs-", lw=2, markersize=8)
    best_k = k_range[silhouettes.index(max(silhouettes))]
    axes[1].axvline(best_k, color="green", linestyle="--",
                    label=f"Mejor K={best_k}")
    axes[1].set_title("Coeficiente de Silhouette por K", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Numero de clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/01_elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Mejor K segun Silhouette: {best_k}")
    return best_k

def plot_clusters_2d(df, labels, x="AnnualIncome", y="SpendingScore"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    unique = sorted(set(labels))
    for i, label in enumerate(unique):
        mask = labels == label
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        lname = f"Cluster {label}" if label != -1 else "Ruido"
        axes[0].scatter(df.loc[mask, x], df.loc[mask, y],
                        c=color, label=lname, alpha=0.7, s=80, edgecolors="white")
    axes[0].set_title("Segmentos de Clientes\n(Ingreso vs Gasto)", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Ingreso anual (k$)")
    axes[0].set_ylabel("Puntuacion de gasto (1-100)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    pca = PCA(n_components=2, random_state=42)
    from sklearn.preprocessing import StandardScaler
    X_pca = pca.fit_transform(StandardScaler().fit_transform(
        df[["Age", "AnnualIncome", "SpendingScore"]]
    ))
    for i, label in enumerate(unique):
        mask = labels == label
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        lname = f"Cluster {label}" if label != -1 else "Ruido"
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, label=lname, alpha=0.7, s=80, edgecolors="white")
    axes[1].set_title(f"Proyeccion PCA 2D\n(Var. explicada: {pca.explained_variance_ratio_.sum():.1%})",
                       fontweight="bold", fontsize=13)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/02_clusters_2d.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_cluster_profiles(df, labels):
    df_plot = df[["Age", "AnnualIncome", "SpendingScore"]].copy()
    df_plot["Cluster"] = [f"Cluster {l}" for l in labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    features = ["Age", "AnnualIncome", "SpendingScore"]
    titles = ["Edad", "Ingreso Anual (k$)", "Puntuacion de Gasto"]

    for ax, feat, title in zip(axes, features, titles):
        data_by_cluster = [df_plot.loc[df_plot["Cluster"] == f"Cluster {l}", feat].values
                           for l in sorted(set(labels))]
        bp = ax.boxplot(data_by_cluster, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], CLUSTER_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_title(f"Distribucion de {title} por Cluster",
                     fontweight="bold", fontsize=11)
        ax.set_xticklabels([f"C{l}" for l in sorted(set(labels))])
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Perfil de Cada Segmento de Clientes",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../outputs/03_cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_dendrogram(X_scaled):
    linked = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linked, truncate_mode="level", p=5,
               color_threshold=6, ax=ax)
    ax.set_title("Dendrograma — Clustering Jerarquico (Ward)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Indice de muestra")
    ax.set_ylabel("Distancia euclidiana")
    ax.axhline(6, color="red", linestyle="--", label="Corte sugerido")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("../outputs/04_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_comparison(metrics_list):
    df_m = pd.DataFrame([m for m in metrics_list if m])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col, title, better in zip(
        axes,
        ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"],
        ["Silhouette (mayor es mejor)", "Davies-Bouldin (menor es mejor)",
         "Calinski-Harabasz (mayor es mejor)"],
        ["max", "min", "max"]
    ):
        best_val = df_m[col].max() if better == "max" else df_m[col].min()
        colors = ["#e74c3c" if v == best_val else "#3498db" for v in df_m[col]]
        ax.bar(df_m["Model"], df_m[col], color=colors, edgecolor="white")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Comparacion de Algoritmos de Clustering",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../outputs/05_algorithm_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()