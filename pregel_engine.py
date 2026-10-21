"""
GraphX-Equivalent Pregel Bulk Synchronous Parallel (BSP) Engine
Executes iterative, message-passing graph-parallel clustering (HP-LPA) matching Apache Spark GraphX Pregel API.
"""

import time
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from heterogeneous_graph import HeterogeneousGraph


class PregelClusteringEngine:
    """
    Simulates Apache Spark GraphX's Pregel computation model:
      - Bulk Synchronous Parallel (BSP) supersteps
      - Vertex Program: vprog(vertex_id, state, message)
      - Send Message: sendMsg(edge_triplet) -> List[(target_vertex_id, message)]
      - Merge Message: mergeMsg(msgA, msgB) -> combined_msg
      - Global barrier synchronization and active vertex tracking
    """
    def __init__(self, graph: HeterogeneousGraph):
        self.graph = graph
        self.superstep_history: List[Dict[str, Any]] = []

    def run_pregel_clustering(
        self,
        max_supersteps: int = 25
    ) -> Dict[str, Any]:
        """
        Executes Heterogeneous Acuity-Weighted Label Propagation (HP-LPA) via Pregel BSP.
        """
        self.superstep_history = []
        start_time = time.time()

        # Step 0: Initialize each vertex with its own ID
        for vid, vdata in self.graph.vertices.items():
            vdata["cluster_label"] = vid
            vdata["confidence"] = 1.0
            vdata["active"] = True

        superstep = 0
        total_messages_exchanged = 0

        while superstep < max_supersteps:
            step_start = time.time()
            messages_pool: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
            messages_sent_in_step = 0

            # -------------------------------------------------------------
            # 1. SEND MESSAGE PHASE: Evaluated across active edges
            # -------------------------------------------------------------
            for src_id, dst_id, weight in self.graph.edges:
                src_attr = self.graph.vertices[src_id]
                dst_attr = self.graph.vertices[dst_id]

                # Propagate when either endpoint changed label in prior superstep
                if src_attr["active"] or dst_attr["active"] or superstep == 0:
                    w_src = weight * src_attr["acuity_weight"]
                    messages_pool[dst_id][src_attr["cluster_label"]] += w_src

                    w_dst = weight * dst_attr["acuity_weight"]
                    messages_pool[src_id][dst_attr["cluster_label"]] += w_dst

                    messages_sent_in_step += 2

            total_messages_exchanged += messages_sent_in_step

            # -------------------------------------------------------------
            # 2. GLOBAL SYNCHRONIZATION BARRIER & VERTEX PROGRAM (vprog)
            # -------------------------------------------------------------
            updated_vertices_count = 0
            active_next_step = 0

            for vid, vdata in self.graph.vertices.items():
                incoming_msgs = messages_pool.get(vid, None)

                if incoming_msgs:
                    # In Pregel LPA, choose label that maximizes neighborhood weight
                    # When multiple labels have identical weight, pick the minimum label ID (deterministic tie-breaker)
                    sorted_labels = sorted(incoming_msgs.items(), key=lambda x: (-x[1], x[0]))
                    best_label = sorted_labels[0][0]
                    total_mass = sum(incoming_msgs.values())
                    confidence = sorted_labels[0][1] / (total_mass + 1e-6)

                    if best_label != vdata["cluster_label"]:
                        vdata["cluster_label"] = best_label
                        vdata["confidence"] = float(confidence)
                        vdata["active"] = True
                        updated_vertices_count += 1
                        active_next_step += 1
                    else:
                        vdata["confidence"] = float(confidence)
                        vdata["active"] = False
                else:
                    vdata["active"] = False

            step_duration_ms = (time.time() - step_start) * 1000
            unique_clusters = len(set(v["cluster_label"] for v in self.graph.vertices.values()))

            step_record = {
                "superstep": superstep,
                "messages_sent": messages_sent_in_step,
                "label_changes": updated_vertices_count,
                "active_vertices": active_next_step,
                "unique_clusters": unique_clusters,
                "duration_ms": round(step_duration_ms, 2)
            }
            self.superstep_history.append(step_record)

            # Check termination
            if updated_vertices_count == 0 and superstep > 0:
                break

            superstep += 1

        total_elapsed_ms = (time.time() - start_time) * 1000

        # Cluster normalization (re-index raw labels to consecutive IDs 0, 1, 2, ...)
        raw_labels = [v["cluster_label"] for v in self.graph.vertices.values()]
        label_map = {raw: idx for idx, raw in enumerate(sorted(set(raw_labels)))}
        for v in self.graph.vertices.values():
            v["assigned_cluster"] = label_map[v["cluster_label"]]

        return {
            "converged_supersteps": superstep + 1,
            "total_messages": total_messages_exchanged,
            "total_time_ms": round(total_elapsed_ms, 2),
            "num_clusters_found": len(label_map),
            "history": self.superstep_history
        }
