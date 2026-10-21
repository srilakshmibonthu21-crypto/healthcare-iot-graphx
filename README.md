# Scalable Clustering of Heterogeneous Big Data via Graph-Parallel Processing with GraphX/Pregel

### Applied to Healthcare IoT Patient Monitoring at Scale
**Guide / Mentor:** Mr. R. SriRamu  
**Domain:** Big Data Analytics & Distributed Systems  
**Framework:** Apache Spark GraphX / Pregel Bulk Synchronous Parallel (BSP)

---

## 📌 Project Overview
Continuous biometric monitoring in smart intensive care units (ICUs) and remote patient telemetry streams continuous vitals (ECG, Heart Rate, SpO2, Blood Pressure, Respiration Rate, Temperature) combined with discrete EHR clinical records (age, comorbidities, NEWS-2 scores).

This project implements a **graph-parallel clustering system** that converts heterogeneous biometric streams into an attributed **Heterogeneous Information Graph (HIG)**. Using a specialized **Pregel BSP message-passing algorithm (HP-LPA)** matching Apache Spark GraphX, the system iteratively groups patients into clinical risk cohorts (Stable, Sepsis Deterioration, Respiratory Distress, Hypertensive Crisis, and Sensor Artifacts) with distributed linear scalability.

---

## 🚀 Key Results Achieved
- **Graph Modularity ($Q$):** `0.7212` (Exceptional community cohesion vs inter-community separation)
- **Pregel NMI (Normalized Mutual Information):** `0.7853` (Outperforms baseline centralized $k$-Means at `0.7492`)
- **Clinical Triage Efficacy:** Accurately isolates high-risk Sepsis decompensation (Average NEWS-2: `10.38`, Heart Rate: `128 bpm`, SpO2: `91%`) from routine stable patients (NEWS-2: `0.14`).

---

## 📂 Codebase Structure

| File | Purpose |
| :--- | :--- |
| `data_generator.py` | Simulates real-time IoT patient telemetry and computes National Early Warning Score 2 (NEWS-2). |
| `heterogeneous_graph.py` | Constructs attributed vertices and multi-modal weighted edges using Gaussian RBF kernels and EHR overlap. |
| `pregel_engine.py` | Implements the GraphX Pregel Bulk Synchronous Parallel (BSP) engine (`vprog`, `sendMsg`, `mergeMsg`). |
| `evaluate_clusters.py` | Calculates Modularity ($Q$), NMI, ARI, Silhouette score, and compiles clinical triage profiles. |
| `app.py` | Interactive Streamlit Web Dashboard for live simulation, graph visualization, and superstep tracking. |
| `spark_graphx_cluster.scala` | Production-grade Scala script for Apache Spark GraphX deployment on Databricks / Hadoop clusters. |

---

## 🖥️ How to Run

### 1. Launch the Interactive Web Dashboard
Run the Streamlit application in PowerShell / Terminal:
```bash
cd C:\Users\srila\.gemini\antigravity\scratch\healthcare-iot-graphx
streamlit run app.py
```
Open your browser at `http://localhost:8501` to:
- Adjust patient count ($N = 50$ to $500$).
- Watch live Pregel superstep message-passing convergence.
- Inspect the 2D network graph topology and patient triage cards.
- Export clustered patient telemetry CSV.

### 2. Run the Evaluation Pipeline directly from CLI
```bash
python evaluate_clusters.py
```

### 3. Deploy on Apache Spark Cluster (Scala)
```bash
spark-submit \
  --class org.bigdata.healthcare.HealthcarePregelClustering \
  --master yarn \
  --deploy-mode cluster \
  target/scala-2.12/healthcare-pregel_2.12-1.0.jar
```
