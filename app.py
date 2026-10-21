"""
Interactive Streamlit Dashboard
Scalable Clustering of Heterogeneous Healthcare IoT Big Data via GraphX/Pregel
Guide: Mr. R. SriRamu
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

from data_generator import generate_healthcare_iot_dataset
from heterogeneous_graph import build_heterogeneous_iot_graph
from pregel_engine import PregelClusteringEngine
from evaluate_clusters import evaluate_clinical_clustering, compute_graph_modularity

st.set_page_config(
    page_title="Healthcare IoT Pregel Clustering",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Scalable Clustering of Heterogeneous Healthcare IoT Big Data")
st.subheader("Graph-Parallel Processing via GraphX / Pregel Bulk Synchronous Parallel (BSP)")
st.caption("Applied to Real-Time ICU Patient Monitoring & Deterioration Triage | Guide: **Mr. R. SriRamu**")

# Sidebar Controls
st.sidebar.header("⚙️ Simulation & Graph Controls")
num_patients = st.sidebar.slider("Number of IoT Patients (V)", min_value=50, max_value=500, value=150, step=25)
k_neighbors = st.sidebar.slider("k-NN Sparsity Factor (Degree)", min_value=4, max_value=16, value=8, step=1)
max_supersteps = st.sidebar.slider("Max Pregel Supersteps", min_value=5, max_value=30, value=15, step=5)
gamma_rbf = st.sidebar.slider("Gaussian RBF Kernel Gamma (γ)", min_value=0.05, max_value=0.50, value=0.15, step=0.05)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)

run_button = st.sidebar.button("🚀 Run Pregel Graph-Parallel Clustering", use_container_width=True)

# Main Execution Flow
if run_button or "df" not in st.session_state:
    with st.spinner("Generating streaming biometric telemetry and clinical EHR records..."):
        df = generate_healthcare_iot_dataset(num_patients=num_patients, random_seed=seed)
        st.session_state["df"] = df

    with st.spinner("Constructing Heterogeneous Information Graph (HIG)..."):
        t0 = time.time()
        g = build_heterogeneous_iot_graph(df, k_neighbors=k_neighbors, gamma_rbf=gamma_rbf)
        t_graph = (time.time() - t0) * 1000
        st.session_state["graph"] = g
        st.session_state["t_graph"] = t_graph

    with st.spinner("Executing Pregel Bulk Synchronous Parallel (BSP) message-passing..."):
        engine = PregelClusteringEngine(g)
        res = engine.run_pregel_clustering(max_supersteps=max_supersteps)
        st.session_state["engine_res"] = res
        st.session_state["eval_res"] = evaluate_clinical_clustering(g, df)

# Retrieve results from session state
df = st.session_state["df"]
g = st.session_state["graph"]
res = st.session_state["engine_res"]
eval_res = st.session_state["eval_res"]

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Patients (Vertices)", len(g.vertices))
col2.metric("Heterogeneous Edges", len(g.edges) // 2)
col3.metric("Pregel Supersteps", f"{res['converged_supersteps']} steps")
col4.metric("Graph Modularity (Q)", f"{eval_res['pregel_modularity_Q']:.4f}")
col5.metric("Pregel NMI Score", f"{eval_res['pregel_nmi']:.4f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Clinical Triage & Cohorts", "🔄 Pregel BSP Execution", "🕸️ Graph Network Topology", "📄 Patient Records"])

with tab1:
    st.subheader("Clinical Triage Stratification & Early Deterioration Warning")
    st.write("Identified cohorts mapped against **National Early Warning Score 2 (NEWS-2)** and biometric telemetry:")

    profiles_df = eval_res["cluster_profiles"]
    
    # Highlight Triage Acuity
    def highlight_acuity(val):
        if "CRITICAL" in str(val):
            return "background-color: #ffcccc; color: #990000; font-weight: bold;"
        elif "MODERATE" in str(val):
            return "background-color: #fff2cc; color: #996600; font-weight: bold;"
        elif "STABLE" in str(val):
            return "background-color: #d9ead3; color: #274e13; font-weight: bold;"
        elif "SENSOR FAULT" in str(val):
            return "background-color: #eeeeee; color: #555555; font-style: italic;"
        return ""

    st.dataframe(
        profiles_df.style.map(highlight_acuity, subset=["Triage Acuity"]),
        use_container_width=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Model Comparison: Pregel vs Centralized k-Means")
        comp_df = pd.DataFrame({
            "Metric": ["Modularity (Q)", "Normalized Mutual Info (NMI)", "Adjusted Rand Index (ARI)"],
            "Proposed Pregel GraphX": [eval_res["pregel_modularity_Q"], eval_res["pregel_nmi"], eval_res["pregel_ari"]],
            "Baseline k-Means": ["N/A (Non-graph)", eval_res["kmeans_baseline_nmi"], eval_res["kmeans_baseline_ari"]]
        })
        st.table(comp_df)

    with col_b:
        st.subheader("Vitals Distribution Across Major Clusters")
        top_clusters = profiles_df.head(5)["Cluster ID"].tolist()
        df_filtered = df[df["patient_id"].apply(lambda pid: g.vertices[pid]["assigned_cluster"] in top_clusters)].copy()
        df_filtered["Cluster"] = df_filtered["patient_id"].apply(lambda pid: f"Cluster {g.vertices[pid]['assigned_cluster']}")
        st.bar_chart(df_filtered.groupby("Cluster")[["heart_rate", "systolic_bp", "news2_score"]].mean())

with tab2:
    st.subheader("Bulk Synchronous Parallel (BSP) Superstep Progression")
    st.write(f"Total Messages Exchanged across Spark cluster: **{res['total_messages']:,}** | Total Execution Time: **{res['total_time_ms']:.2f} ms**")

    history_df = pd.DataFrame(res["history"])
    st.dataframe(history_df, use_container_width=True)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Pregel Superstep')
    ax1.set_ylabel('Messages Exchanged', color=color)
    ax1.plot(history_df['superstep'], history_df['messages_sent'], marker='o', color=color, linewidth=2, label="Messages Sent")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Active Vertices / Label Updates', color=color)
    ax2.plot(history_df['superstep'], history_df['label_changes'], marker='s', color=color, linestyle='--', linewidth=2, label="Label Updates")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Heterogeneous Information Graph Visualization")
    st.write("Visualized sample sub-network showing patient nodes colored by assigned triage cluster:")

    G_nx = nx.Graph()
    sample_nodes = list(g.vertices.keys())[:min(80, len(g.vertices))]
    sample_set = set(sample_nodes)

    for vid in sample_nodes:
        vdata = g.vertices[vid]
        G_nx.add_node(vid, cluster=vdata["assigned_cluster"], news2=vdata["news2_score"])

    for u, v, w in g.edges:
        if u in sample_set and v in sample_set and u < v:
            G_nx.add_edge(u, v, weight=w)

    fig_net, ax_net = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G_nx, seed=42, k=0.35)
    
    node_colors = [G_nx.nodes[n]["cluster"] for n in G_nx.nodes]
    nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, cmap=plt.cm.tab10, node_size=180, alpha=0.9, ax=ax_net)
    nx.draw_networkx_edges(G_nx, pos, alpha=0.25, ax=ax_net)
    ax_net.set_title("Patient Similarity Graph Topology (Sample 80 Vertices)")
    ax_net.axis("off")
    st.pyplot(fig_net)

with tab4:
    st.subheader("Raw IoT Telemetry & Cluster Assignment Data")
    display_df = df.copy()
    display_df["Assigned_Cluster"] = display_df["patient_id"].apply(lambda pid: g.vertices[pid]["assigned_cluster"])
    st.dataframe(display_df, use_container_width=True)

    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Clustered Patient Telemetry CSV",
        data=csv,
        file_name="healthcare_iot_pregel_clusters.csv",
        mime="text/csv"
    )
