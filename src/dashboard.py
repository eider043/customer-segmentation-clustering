"""
Dashboard narrativo — Segmentacion de Clientes
Autor: Eider
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Estilos ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 10px; }
    h1 { color: #e74c3c; }
    h2 { color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
    h3 { color: #2ecc71; }
    .insight-box {
        background-color: #1e2130;
        border-left: 4px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

# ── Carga de datos ───────────────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("../data/customers_clustered.csv")
    return df

@st.cache_data
def compute_elbow(X_scaled, k_max=10):
    inertias, silhouettes = [], []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return list(range(2, k_max + 1)), inertias, silhouettes

df = load()
X = df[["Age", "AnnualIncome", "SpendingScore"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("Configuracion")
n_clusters = st.sidebar.slider("Numero de clusters (K-Means)", 2, 8, 5)
feature_x = st.sidebar.selectbox("Eje X", ["AnnualIncome", "Age", "SpendingScore"], index=0)
feature_y = st.sidebar.selectbox("Eje Y", ["SpendingScore", "Age", "AnnualIncome"], index=0)

km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)
df["Cluster"] = [f"Cluster {l}" for l in labels]
sil_score = silhouette_score(X_scaled, labels)

# ── Header ───────────────────────────────────────────────────────────
st.title("Customer Segmentation — Clustering Analysis")
st.markdown("**Una historia sobre como los datos revelan patrones ocultos en el comportamiento de clientes**")
st.markdown("---")

# ── Capitulo 1: El negocio ───────────────────────────────────────────
st.header("Capitulo 1 — El Problema de Negocio")
st.markdown("""
<div class='insight-box'>
Un centro comercial quiere entender mejor a sus clientes para personalizar estrategias de marketing.
Tiene datos de 200 clientes: edad, ingreso anual y puntuacion de gasto. La pregunta es:
<b>existen grupos naturales de clientes con comportamientos similares?</b>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total clientes", len(df))
col2.metric("Variables analizadas", 3)
col3.metric("Clusters encontrados", n_clusters)
col4.metric("Silhouette Score", f"{sil_score:.4f}")

# ── Capitulo 2: EDA ──────────────────────────────────────────────────
st.header("Capitulo 2 — Exploracion de los Datos")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="Age", color_discrete_sequence=["#3498db"],
                       nbins=20, title="Distribucion de Edad",
                       template="plotly_dark")
    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df, x="AnnualIncome", color_discrete_sequence=["#2ecc71"],
                       nbins=20, title="Distribucion de Ingreso Anual (k$)",
                       template="plotly_dark")
    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="SpendingScore", color_discrete_sequence=["#e74c3c"],
                       nbins=20, title="Distribucion de Puntuacion de Gasto",
                       template="plotly_dark")
    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter_matrix(df[["Age", "AnnualIncome", "SpendingScore"]],
                             color_discrete_sequence=["#f39c12"],
                             template="plotly_dark",
                             title="Matriz de dispersion entre variables")
    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class='insight-box'>
<b>Insight EDA:</b> La distribucion de ingresos es relativamente uniforme entre 15k y 137k dolares.
La puntuacion de gasto varia ampliamente (1-99), lo que sugiere segmentos muy distintos.
No hay correlacion lineal evidente entre ingreso y gasto — un hallazgo clave para el clustering.
</div>
""", unsafe_allow_html=True)

# ── Capitulo 3: Encontrando K optimo ─────────────────────────────────
st.header("Capitulo 3 — Cuantos Grupos Existen?")

k_range, inertias, silhouettes = compute_elbow(X_scaled)

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Metodo del Codo", "Coeficiente de Silhouette"))

fig.add_trace(go.Scatter(x=k_range, y=inertias, mode="lines+markers",
                          marker=dict(size=10, color="#e74c3c"),
                          line=dict(color="#e74c3c", width=2),
                          name="Inercia"), row=1, col=1)

fig.add_trace(go.Scatter(x=k_range, y=silhouettes, mode="lines+markers",
                          marker=dict(size=10, color="#3498db"),
                          line=dict(color="#3498db", width=2),
                          name="Silhouette"), row=1, col=2)

best_k_sil = k_range[silhouettes.index(max(silhouettes))]
fig.add_vline(x=best_k_sil, line_dash="dash", line_color="#2ecc71",
              annotation_text=f"K optimo={best_k_sil}", row=1, col=2)

fig.update_layout(template="plotly_dark", height=400,
                  plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                  showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
<div class='insight-box'>
<b>Conclusion:</b> El metodo del codo sugiere K=5 como punto de inflexion donde
la reduccion de inercia se estabiliza. El coeficiente de Silhouette confirma K={best_k_sil}
como optimo (score mas alto = clusters mas cohesionados y separados).
</div>
""", unsafe_allow_html=True)

# ── Capitulo 4: Resultados del Clustering ────────────────────────────
st.header("Capitulo 4 — Los Segmentos Descubiertos")

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(df, x=feature_x, y=feature_y, color="Cluster",
                     color_discrete_sequence=COLORS,
                     title=f"Segmentos: {feature_x} vs {feature_y}",
                     template="plotly_dark", size_max=12,
                     hover_data=["Age", "AnnualIncome", "SpendingScore", "Gender"])
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_pca["Cluster"] = df["Cluster"]
    var = pca.explained_variance_ratio_

    fig = px.scatter(df_pca, x="PC1", y="PC2", color="Cluster",
                     color_discrete_sequence=COLORS,
                     title=f"Proyeccion PCA 2D (Var. explicada: {var.sum():.1%})",
                     template="plotly_dark")
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    fig.update_layout(
        plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
        xaxis_title=f"PC1 ({var[0]:.1%})",
        yaxis_title=f"PC2 ({var[1]:.1%})"
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Capitulo 5: Perfiles de segmentos ────────────────────────────────
st.header("Capitulo 5 — Perfil de Cada Segmento")

profile = df.groupby("Cluster")[["Age", "AnnualIncome", "SpendingScore"]].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index().values
profile["Gender_F%"] = df.groupby("Cluster")["Gender"].apply(
    lambda x: (x == "Female").mean() * 100
).round(1).values

st.dataframe(profile.style.background_gradient(cmap="Blues"), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig = px.box(df, x="Cluster", y="SpendingScore", color="Cluster",
                 color_discrete_sequence=COLORS,
                 title="Distribucion de Gasto por Segmento",
                 template="plotly_dark")
    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df, x="Cluster", y="AnnualIncome", color="Cluster",
                 color_discrete_sequence=COLORS,
                 title="Distribucion de Ingreso por Segmento",
                 template="plotly_dark")
    fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

fig = px.scatter_3d(df, x="Age", y="AnnualIncome", z="SpendingScore",
                     color="Cluster", color_discrete_sequence=COLORS,
                     title="Vista 3D de Segmentos",
                     template="plotly_dark", opacity=0.85)
fig.update_layout(plot_bgcolor="#1e2130", paper_bgcolor="#1e2130", height=600)
st.plotly_chart(fig, use_container_width=True)

# ── Capitulo 6: Interpretacion ───────────────────────────────────────
st.header("Capitulo 6 — Interpretacion de Segmentos")

interpretaciones = {
    "Cluster 0": ("Jovenes gastadores", "Ingresos bajos pero alto gasto. Clientes impulsivos o con prioridad en experiencias."),
    "Cluster 1": ("Conservadores maduros", "Ingresos altos pero gasto bajo. Ahorradores o con preferencias fuera del mall."),
    "Cluster 2": ("Objetivo principal", "Alto ingreso y alto gasto. El segmento mas valioso para estrategias premium."),
    "Cluster 3": ("Clientes promedio", "Ingreso y gasto moderados. Segmento mas grande y heterogeneo."),
    "Cluster 4": ("Sensibles al precio", "Ingreso bajo y gasto bajo. Requieren promociones y descuentos para activarse."),
}

cols = st.columns(len(interpretaciones))
for col, (cluster, (nombre, desc)) in zip(cols, interpretaciones.items()):
    with col:
        st.markdown(f"**{cluster}**")
        st.markdown(f"*{nombre}*")
        st.caption(desc)

# ── Capitulo 7: Comparacion de algoritmos ────────────────────────────
st.header("Capitulo 7 — Comparacion de Algoritmos")

try:
    metrics_df = pd.read_csv("../outputs/clustering_metrics.csv")
    col1, col2, col3 = st.columns(3)
    for col, metric, better in zip(
        [col1, col2, col3],
        ["Silhouette", "Davies_Bouldin", "Calinski_Harabasz"],
        ["max", "min", "max"]
    ):
        with col:
            best = metrics_df[metric].max() if better == "max" else metrics_df[metric].min()
            colors = ["#e74c3c" if v == best else "#3498db" for v in metrics_df[metric]]
            fig = go.Figure(go.Bar(
                x=metrics_df["Model"], y=metrics_df[metric],
                marker_color=colors, text=metrics_df[metric].round(4),
                textposition="outside"
            ))
            fig.update_layout(
                title=metric, template="plotly_dark", height=300,
                plot_bgcolor="#1e2130", paper_bgcolor="#1e2130",
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
except FileNotFoundError:
    st.warning("Ejecuta primero main.py para generar las metricas comparativas.")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**Eider** — Cientifico de Datos |
[GitHub](https://github.com/eider043) |
[Fiverr](https://www.fiverr.com/eiderdatadriven)
""")