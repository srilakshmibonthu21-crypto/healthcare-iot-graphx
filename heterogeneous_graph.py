"""
Heterogeneous Information Graph (HIG) Constructor for Healthcare IoT
Constructs attributed vertices and multi-modal weighted edges based on biometric signals and clinical profiles.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors


class HeterogeneousGraph:
    """
    Graph representation matching Apache Spark GraphX RDD[Vertex] and RDD[Edge] abstractions.
    """
    def __init__(self):
        self.vertices: Dict[int, Dict[str, Any]] = {}
        self.edges: List[Tuple[int, int, float]] = []
        self.adjacency: Dict[int, List[Tuple[int, float]]] = {}

    def add_vertex(self, vertex_id: int, attributes: Dict[str, Any]):
        self.vertices[vertex_id] = attributes
        if vertex_id not in self.adjacency:
            self.adjacency[vertex_id] = []

    def add_edge(self, src: int, dst: int, weight: float):
        self.edges.append((src, dst, weight))
        self.edges.append((dst, src, weight))
        self.adjacency[src].append((dst, weight))
        self.adjacency[dst].append((src, weight))


def build_heterogeneous_iot_graph(
    df: pd.DataFrame,
    vital_cols: List[str] = None,
    k_neighbors: int = 8,
    gamma_rbf: float = 0.15,
    sim_threshold: float = 0.30
) -> HeterogeneousGraph:
    """
    Constructs a Heterogeneous Information Graph (HIG) using Gaussian RBF biometric kernel
    and clinical comorbidity overlap.
    """
    if vital_cols is None:
        vital_cols = ["heart_rate", "spo2", "systolic_bp", "resp_rate", "temperature"]

    graph = HeterogeneousGraph()

    # 1. Standardize continuous vital signs
    scaler = StandardScaler()
    scaled_vitals = scaler.fit_transform(df[vital_cols].values)

    # 2. Add Vertices
    for idx, row in df.iterrows():
        pid = int(row["patient_id"])
        news2 = int(row["news2_score"])
        
        # Clinical acuity weight (higher NEWS-2 gives higher authority in community propagation)
        acuity_weight = 1.0 + (news2 * 0.15)

        vertex_data = {
            "vertex_id": pid,
            "patient_id": pid,
            "vitals": row[vital_cols].to_dict(),
            "comorbidity": row["comorbidity"],
            "news2_score": news2,
            "true_cohort": row["true_cohort"],
            "cluster_label": pid,
            "confidence": 1.0,
            "acuity_weight": acuity_weight,
            "active": True
        }
        graph.add_vertex(pid, vertex_data)

    # 3. Compute Gaussian RBF Similarity Matrix over scaled vitals
    rbf_sim = rbf_kernel(scaled_vitals, gamma=gamma_rbf)

    # 4. Construct k-Nearest Neighbors Sparse Graph
    n_pts = len(df)
    k_actual = min(k_neighbors + 1, n_pts)
    nbrs = NearestNeighbors(n_neighbors=k_actual, metric="euclidean").fit(scaled_vitals)
    _, indices = nbrs.kneighbors(scaled_vitals)

    patient_ids = df["patient_id"].values
    comorbidities = df["comorbidity"].values
    seen_pairs = set()

    for i in range(n_pts):
        u_id = int(patient_ids[i])
        u_comorb = comorbidities[i]

        for rank in range(1, k_actual):
            j = indices[i, rank]
            v_id = int(patient_ids[j])
            
            pair = tuple(sorted((u_id, v_id)))
            if pair in seen_pairs or u_id == v_id:
                continue
            seen_pairs.add(pair)

            sim_vital = float(rbf_sim[i, j])
            v_comorb = comorbidities[j]
            sim_clinical = 1.0 if (u_comorb == v_comorb and u_comorb != "None") else 0.2

            # Heterogeneous fusion weight
            weight = (0.85 * sim_vital) + (0.15 * sim_clinical)

            if weight >= sim_threshold:
                graph.add_edge(u_id, v_id, round(float(weight), 4))

    return graph
