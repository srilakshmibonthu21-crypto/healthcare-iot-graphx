"""
Clustering Evaluation and Clinical Triage Benchmarking
Calculates Graph Modularity (Q), NMI, ARI, Silhouette Score, and Clinical Risk Classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from heterogeneous_graph import HeterogeneousGraph


def compute_graph_modularity(graph: HeterogeneousGraph) -> float:
    """
    Computes Newman-Girvan Modularity (Q) for the assigned clusters:
    Q = 1/(2m) * sum_ij [ A_ij - (k_i * k_j)/(2m) ] * delta(c_i, c_j)
    """
    # Total edge weight (2m for undirected graph)
    total_m = sum(w for _, _, w in graph.edges)
    if total_m == 0:
        return 0.0

    # Vertex degree strengths
    degrees = {vid: 0.0 for vid in graph.vertices}
    for u, v, w in graph.edges:
        degrees[u] += w

    cluster_assignments = {vid: vdata["assigned_cluster"] for vid, vdata in graph.vertices.items()}

    # Compute modularity
    q_score = 0.0
    for u, v, w in graph.edges:
        if cluster_assignments[u] == cluster_assignments[v]:
            q_score += (w - (degrees[u] * degrees[v]) / total_m)

    return round(float(q_score / total_m), 4)


def evaluate_clinical_clustering(
    graph: HeterogeneousGraph,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates Pregel HP-LPA clusters against ground truth clinical cohorts and k-Means baseline.
    """
    vital_cols = ["heart_rate", "spo2", "systolic_bp", "resp_rate", "temperature"]
    X = df[vital_cols].values

    true_cohorts = df["true_cohort"].values
    predicted_clusters = np.array([graph.vertices[pid]["assigned_cluster"] for pid in df["patient_id"]])

    # 1. Graph Metrics
    modularity_q = compute_graph_modularity(graph)

    # 2. Information Theoretic Metrics
    nmi = normalized_mutual_info_score(true_cohorts, predicted_clusters)
    ari = adjusted_rand_score(true_cohorts, predicted_clusters)

    # 3. Geometric Silhouette
    if len(set(predicted_clusters)) > 1:
        sil = silhouette_score(X, predicted_clusters)
    else:
        sil = 0.0

    # 4. Compare with Baseline k-Means (k=5)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
    baseline_nmi = normalized_mutual_info_score(true_cohorts, kmeans.labels_)
    baseline_ari = adjusted_rand_score(true_cohorts, kmeans.labels_)

    # 5. Cohort Acuity Profile per Cluster
    cluster_profiles = []
    for c_id in sorted(set(predicted_clusters)):
        mask = (predicted_clusters == c_id)
        subset_df = df[mask]
        
        avg_hr = subset_df["heart_rate"].mean()
        avg_spo2 = subset_df["spo2"].mean()
        avg_sbp = subset_df["systolic_bp"].mean()
        avg_rr = subset_df["resp_rate"].mean()
        avg_news2 = subset_df["news2_score"].mean()
        dominant_cohort = subset_df["true_cohort"].mode().iloc[0] if len(subset_df) > 0 else "Unknown"

        # Determine clinical risk tier
        if avg_news2 >= 7.0 or dominant_cohort == "Sepsis_Risk":
            triage_tier = "CRITICAL (Red Alert - Sepsis/Decompensation)"
        elif avg_news2 >= 4.5 or dominant_cohort in ["Respiratory_Distress", "Hypertensive_Crisis"]:
            triage_tier = "MODERATE (Yellow Alert - Elevated Acuity)"
        elif dominant_cohort == "Sensor_Fault":
            triage_tier = "SENSOR FAULT (Gray - Equipment Inspection)"
        else:
            triage_tier = "STABLE (Green - Routine Telemetry)"

        cluster_profiles.append({
            "Cluster ID": c_id,
            "Patient Count": len(subset_df),
            "Triage Acuity": triage_tier,
            "Dominant Pattern": dominant_cohort,
            "Avg NEWS-2": round(avg_news2, 2),
            "Avg HR": round(avg_hr, 1),
            "Avg SpO2 (%)": round(avg_spo2, 1),
            "Avg SBP (mmHg)": round(avg_sbp, 1),
            "Avg RR (bpm)": round(avg_rr, 1),
        })

    return {
        "pregel_modularity_Q": modularity_q,
        "pregel_nmi": round(float(nmi), 4),
        "pregel_ari": round(float(ari), 4),
        "pregel_silhouette": round(float(sil), 4),
        "kmeans_baseline_nmi": round(float(baseline_nmi), 4),
        "kmeans_baseline_ari": round(float(baseline_ari), 4),
        "cluster_profiles": pd.DataFrame(cluster_profiles)
    }


if __name__ == "__main__":
    from data_generator import generate_healthcare_iot_dataset
    from heterogeneous_graph import build_heterogeneous_iot_graph
    from pregel_engine import PregelClusteringEngine

    df = generate_healthcare_iot_dataset(num_patients=200)
    g = build_heterogeneous_iot_graph(df)
    engine = PregelClusteringEngine(g)
    engine.run_pregel_clustering()
    eval_res = evaluate_clinical_clustering(g, df)
    print("\nEvaluation Results:")
    print(f"Modularity (Q): {eval_res['pregel_modularity_Q']}")
    print(f"Pregel NMI: {eval_res['pregel_nmi']} vs Baseline k-Means NMI: {eval_res['kmeans_baseline_nmi']}")
    print(f"Pregel ARI: {eval_res['pregel_ari']} vs Baseline k-Means ARI: {eval_res['kmeans_baseline_ari']}")
    print("\nCluster Profiles:")
    print(eval_res["cluster_profiles"].to_string())
