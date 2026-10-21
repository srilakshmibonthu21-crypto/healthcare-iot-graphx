/**
 * Scalable Clustering of Heterogeneous Big Data via Graph-Parallel Processing
 * Apache Spark GraphX / Pregel Implementation for Healthcare IoT Patient Monitoring
 * 
 * Guide: Mr. R. SriRamu
 * Build: sbt package / spark-submit --class HealthcarePregelClustering target/scala-2.12/healthcare-pregel_2.12-1.0.jar
 */

package org.bigdata.healthcare

import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

// 1. Domain Entities & State Structures
case class PatientState(
  patientId: Long,
  clusterLabel: Long,
  acuityWeight: Double,
  news2Score: Int,
  heartRate: Double,
  spo2: Double,
  systolicBp: Double,
  respRate: Double,
  temperature: Double,
  isActive: Boolean = true
)

case class EdgeWeight(
  compositeSimilarity: Double,
  biometricSim: Double,
  clinicalSim: Double
)

object HealthcarePregelClustering {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("HealthcareIoT_GraphX_Pregel_Clustering")
      .getOrCreate()

    val sc = spark.sparkContext
    import spark.implicits._

    println(">>> Starting GraphX Heterogeneous IoT Patient Graph Construction...")

    // 2. Mock RDD Creation (or read from Parquet / Delta Lake / Kafka stream)
    val rawPatients = Seq(
      PatientState(1001L, 1001L, 1.0, 0, 72.0, 98.0, 120.0, 15.0, 36.8),
      PatientState(1002L, 1002L, 1.2, 1, 74.0, 97.0, 122.0, 16.0, 36.9),
      PatientState(1003L, 1003L, 2.5, 9, 130.0, 88.0, 78.0, 28.0, 39.4), // Sepsis alert
      PatientState(1004L, 1004L, 2.8, 11, 135.0, 86.0, 75.0, 30.0, 39.6), // Sepsis alert
      PatientState(1005L, 1005L, 1.8, 4, 96.0, 95.0, 195.0, 18.0, 37.1)   // Hypertensive
    )

    val verticesRDD: RDD[(VertexId, PatientState)] = sc.parallelize(rawPatients.map(p => (p.patientId, p)))

    val edgesRDD: RDD[Edge[Double]] = sc.parallelize(Seq(
      Edge(1001L, 1002L, 0.92), // Stable - Stable similarity
      Edge(1003L, 1004L, 0.95), // Sepsis - Sepsis similarity
      Edge(1002L, 1005L, 0.40)  // Cross-cohort weak similarity
    ))

    // 3. Graph Creation with 2D Vertex-Cut Partitioning for shuffle minimization
    val initialGraph: Graph[PatientState, Double] = Graph(verticesRDD, edgesRDD)
      .partitionBy(PartitionStrategy.EdgePartition2D)

    println(s">>> Graph initialized with ${initialGraph.vertices.count()} vertices and ${initialGraph.edges.count()} edges.")

    // 4. Pregel Bulk Synchronous Parallel (BSP) Computation
    val maxIterations = 20
    val initialMessage = Map.empty[Long, Double] // Map[ClusterLabel, AccumulatedNeighborhoodWeight]

    val clusteredGraph = initialGraph.pregel(
      initialMsg = initialMessage,
      maxIterations = maxIterations,
      activeDirection = EdgeDirection.Either
    )(
      // -------------------------------------------------------------
      // vprog: Vertex Program executed on each superstep
      // -------------------------------------------------------------
      vprog = (id: VertexId, state: PatientState, incomingMsg: Map[Long, Double]) => {
        if (incomingMsg.isEmpty) {
          state
        } else {
          // Select community label with maximum accumulated affinity weight
          val (dominantLabel, _) = incomingMsg.maxBy(_._2)
          if (dominantLabel != state.clusterLabel) {
            state.copy(clusterLabel = dominantLabel, isActive = true)
          } else {
            state.copy(isActive = false)
          }
        }
      },

      // -------------------------------------------------------------
      // sendMsg: Message dispatch across edge triplets
      // -------------------------------------------------------------
      sendMsg = (triplet: EdgeTriplet[PatientState, Double]) => {
        val src = triplet.srcAttr
        val dst = triplet.dstAttr
        val edgeWeight = triplet.attr

        // Send messages only when an endpoint changed state or labels differ
        if (src.isActive || dst.isActive) {
          val msgToDst = Map(src.clusterLabel -> edgeWeight * src.acuityWeight)
          val msgToSrc = Map(dst.clusterLabel -> edgeWeight * dst.acuityWeight)
          Iterator((triplet.dstId, msgToDst), (triplet.srcId, msgToSrc))
        } else {
          Iterator.empty
        }
      },

      // -------------------------------------------------------------
      // mergeMsg: Aggregates messages arriving at the same vertex
      // -------------------------------------------------------------
      mergeMsg = (msgA: Map[Long, Double], msgB: Map[Long, Double]) => {
        (msgA.keys ++ msgB.keys).map { label =>
          label -> (msgA.getOrElse(label, 0.0) + msgB.getOrElse(label, 0.0))
        }.toMap
      }
    )

    // 5. Output Cluster Summaries & Clinical Triage Cohorts
    println(">>> Pregel Clustering Complete. Inspecting final cluster assignments:")
    clusteredGraph.vertices.collect().foreach { case (id, state) =>
      println(s"Patient ID: $id | Assigned Cluster: ${state.clusterLabel} | NEWS-2: ${state.news2Score} | HR: ${state.heartRate} bpm | SpO2: ${state.spo2}%")
    }

    spark.stop()
  }
}
