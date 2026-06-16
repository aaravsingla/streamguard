# Author: Aarav Singla

import json
import os
import time
from joblib import load
import logging
from multiprocessing import Process

import numpy as np
import requests

from streaming.utils import create_producer, create_consumer
from streaming.drift_detector import DriftDetector
from streaming.metrics import (
    ANOMALIES_DETECTED,
    EVENTS_PROCESSED,
    PSI_VALUE,
    DRIFT_TRIGGERED,
    EVENT_PROCESSING_LATENCY,
    clear_multiproc_dir,
    start_metrics_server,
)
from settings import (TRANSACTIONS_TOPIC, TRANSACTIONS_CONSUMER_GROUP, ANOMALIES_TOPIC,
                      NUM_PARTITIONS, ALERT_SERVICE_URL)

model_path = os.path.abspath('../model/isolation_forest.joblib')

# Port the Prometheus metrics endpoint listens on (scraped by the Prometheus
# container — see docker-compose.yml / prometheus/prometheus.yml).
METRICS_PORT = 8000

# How many incoming transactions to accumulate before running a PSI check.
# PSI needs a *distribution*, not a single point, so we score in batches.
DRIFT_BATCH_SIZE = 500


def build_drift_detector():
    """Rebuild the training feature distribution the model was fit on.

    The Isolation Forest in model/train.py is trained on this exact synthetic
    data, so we regenerate it here to use as the PSI *reference* (expected)
    distribution. In a production setup you'd persist these stats alongside
    the model instead of regenerating them.
    """
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(500, 2)
    X_train = np.r_[X + 2, X - 2]
    return DriftDetector.from_training_data(X_train, feature_names=["feature_0", "feature_1"])


def send_alert(event_id, entity_id, anomaly_score, psi_value, timestamp):
    """POST an anomaly to the FastAPI alerting service.

    Network/alerting failures must NOT take down the consumer, so any error is
    logged as a warning and swallowed — dropping an alert is preferable to
    halting stream processing.
    """
    payload = {
        "event_id": str(event_id),
        "entity_id": str(entity_id),
        "anomaly_score": float(anomaly_score),
        "psi_value": None if psi_value is None else float(psi_value),
        "timestamp": timestamp,
    }
    try:
        requests.post(ALERT_SERVICE_URL, json=payload, timeout=1)
    except requests.RequestException as e:
        logging.warning("Alert service call failed: {}".format(e))


def detect():
    consumer = create_consumer(topic=TRANSACTIONS_TOPIC, group_id=TRANSACTIONS_CONSUMER_GROUP)

    producer = create_producer()

    clf = load(model_path)

    # PSI drift detector seeded with the model's training distribution.
    drift_detector = build_drift_detector()
    # Buffer of feature rows used to evaluate drift over a batch.
    feature_buffer = []
    # Most recent PSI reading, attached to outgoing alerts for context.
    last_psi = None

    while True:
        message = consumer.poll(timeout=50)
        if message is None:
            continue
        if message.error():
            logging.error("Consumer error: {}".format(message.error()))
            continue

        # Time the full per-event processing path (decode -> predict -> route).
        # The histogram captures pipeline latency so we can alert if consumers
        # start lagging behind the stream.
        with EVENT_PROCESSING_LATENCY.time():
            # Message that came from producer
            record = json.loads(message.value().decode('utf-8'))
            data = record["data"]

            # Count every event we consume — throughput + denominator for the
            # anomaly ratio.
            EVENTS_PROCESSED.inc()

            prediction = clf.predict(data)

            # If an anomaly comes in, send it to anomalies topic
            if prediction[0] == -1:
                # Headline metric: a flagged anomaly.
                ANOMALIES_DETECTED.inc()

                score = clf.score_samples(data)
                record["score"] = np.round(score, 3).tolist()

                # Fire an external alert (non-blocking on failure). Done before
                # the record is re-encoded to bytes so we still have the dict.
                send_alert(
                    event_id=record["id"],
                    entity_id=record.get("id"),
                    anomaly_score=float(np.round(score[0], 3)),
                    psi_value=last_psi,
                    timestamp=record.get("current_time"),
                )

                _id = str(record["id"])
                record = json.dumps(record).encode("utf-8")

                producer.produce(topic=ANOMALIES_TOPIC,
                                 value=record)
                producer.flush()

        # --- Concept drift monitoring -------------------------------- #
        # Collect the SAME features the Isolation Forest scores on, then run
        # a PSI check once we have a full batch's worth of observations.
        feature_buffer.extend(data)
        if len(feature_buffer) >= DRIFT_BATCH_SIZE:
            batch_features = np.array(feature_buffer)
            result = drift_detector.compute_psi(batch_features)

            # Publish the latest PSI per feature so Grafana can show the
            # current drift level at a glance.
            for feature_name, psi in result["psi"].items():
                if psi is not None:
                    PSI_VALUE.labels(feature_name=feature_name).set(psi)

            # Keep the worst (max) PSI to attach to subsequent alerts.
            psi_readings = [v for v in result["psi"].values() if v is not None]
            if psi_readings:
                last_psi = max(psi_readings)

            if result["triggered"]:
                DRIFT_TRIGGERED.inc()
                print("DRIFT DETECTED — PSI: {}".format(result["psi"]))
            feature_buffer = []

        # consumer.commit() # Uncomment to process all messages, not just new ones

    consumer.close()


def main():
    # Wipe stale multiprocess metric files from any previous run, then expose
    # the aggregated /metrics endpoint ONCE in the parent process before the
    # per-partition consumers are forked.
    clear_multiproc_dir()
    start_metrics_server(METRICS_PORT)

    # One consumer per partition
    processes = []
    for _ in range(NUM_PARTITIONS):
        p = Process(target=detect)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# Guard so that on 'spawn' start methods (macOS/Windows) re-importing this
# module in a child does not recursively spawn more consumers.
if __name__ == "__main__":
    main()
