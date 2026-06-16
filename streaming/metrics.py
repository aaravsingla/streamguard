# Author: Aarav Singla

"""
Prometheus instrumentation for the StreamGuard anomaly-detection consumer.

WHY MULTIPROCESS MODE?
----------------------
anomalies_detector.py forks one consumer process per Kafka partition
(NUM_PARTITIONS). If each child started its own HTTP server they'd fight over
port 8000 and, worse, every child would keep its own private counters so the
numbers Prometheus scraped would be wrong.

prometheus_client solves this with "multiprocess mode": every process writes
its samples into a shared directory (PROMETHEUS_MULTIPROC_DIR) and a single
HTTP endpoint in the parent aggregates them at scrape time. That gives one
correct, fleet-wide view of the consumers.
"""

import glob
import os
import tempfile

# This MUST be set before the prometheus_client metric objects are imported,
# otherwise they are created in single-process mode. We default to a temp dir
# shared by the parent and all forked children.
MULTIPROC_DIR = os.environ.setdefault(
    "PROMETHEUS_MULTIPROC_DIR",
    os.path.join(tempfile.gettempdir(), "streamguard_prometheus"),
)
os.makedirs(MULTIPROC_DIR, exist_ok=True)

from prometheus_client import (  # noqa: E402  (import after env var is set)
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    start_http_server,
    multiprocess,
)

# --------------------------------------------------------------------------- #
# Metric definitions
#
# NOTE: prometheus_client appends "_total" to Counter names automatically, so
# Counter("anomalies_detected") is exposed as "anomalies_detected_total".
# --------------------------------------------------------------------------- #

# How many anomalies the Isolation Forest has flagged. This is the headline
# number for an anomaly-detection system: a sudden spike can mean a real attack
# / fault, or a broken model. Watching its RATE over time is what matters.
ANOMALIES_DETECTED = Counter(
    "anomalies_detected",
    "Total number of transactions classified as anomalies (prediction == -1).",
)

# Every event pulled off the transactions topic, anomalous or not. Combined
# with anomalies_detected it gives the anomaly RATIO, and on its own it is the
# consumer throughput / liveness signal — if it stops climbing, ingestion or a
# consumer has stalled.
EVENTS_PROCESSED = Counter(
    "events_processed",
    "Total number of events consumed and scored by the model.",
)

# Current Population Stability Index per feature. Labelled by feature so each
# input gets its own line. This is the concept-drift signal: when the live data
# distribution drifts away from training data, PSI climbs and the model's
# predictions become less trustworthy even if no anomalies are firing.
# 'mostrecent' tells the multiprocess collector to expose the latest value set
# by any process (a gauge is a point-in-time reading, not something to sum).
PSI_VALUE = Gauge(
    "psi_value",
    "Current PSI (Population Stability Index) per feature.",
    ["feature_name"],
    multiprocess_mode="mostrecent",
)

# How many times a drift event crossed the PSI threshold. A rising count is the
# cue to investigate/retrain — it separates "the world changed" (drift) from
# "the model found outliers" (anomalies), which are very different problems.
DRIFT_TRIGGERED = Counter(
    "drift_triggered",
    "Total number of times PSI exceeded the drift threshold.",
)

# Wall-clock time to process a single event (poll -> predict -> route). This is
# the latency SLO for the pipeline: if the histogram's high percentiles creep
# up, the consumer is falling behind real time and alerts will be delayed.
EVENT_PROCESSING_LATENCY = Histogram(
    "event_processing_latency_seconds",
    "Time spent processing a single event, in seconds.",
    # Buckets tuned for sub-second, high-throughput event handling.
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)


def clear_multiproc_dir():
    """Remove stale .db files so a fresh run doesn't inherit old counts.

    Call this once in the parent BEFORE forking the consumer processes.
    """
    for db_file in glob.glob(os.path.join(MULTIPROC_DIR, "*.db")):
        try:
            os.remove(db_file)
        except OSError:
            pass


def start_metrics_server(port=8000):
    """Expose aggregated multiprocess metrics on http://0.0.0.0:<port>/metrics.

    Uses a dedicated registry wired to the MultiProcessCollector so the values
    served are the SUM/most-recent across every forked consumer, not just this
    process.
    """
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    start_http_server(port, registry=registry)
