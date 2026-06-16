# Author: Aarav Singla
# Lightweight FastAPI alerting service for StreamGuard anomalies.
import csv
import os
import time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="StreamGuard Alert Service")

ALERTS_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "logs", "alerts.csv"))
CSV_HEADER = ["received_at", "event_id", "entity_id", "anomaly_score", "psi_value", "timestamp"]


class Alert(BaseModel):
    event_id: str
    entity_id: str
    anomaly_score: float
    psi_value: Optional[float] = None
    timestamp: str


def _ensure_csv():
    os.makedirs(os.path.dirname(ALERTS_CSV), exist_ok=True)
    if not os.path.exists(ALERTS_CSV):
        with open(ALERTS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


# Receives an anomaly alert and persists it to a CSV audit log, returning how
# long handling took. In production this would fan out to PagerDuty / a Slack
# webhook / an incident queue instead of appending to a local file.
@app.post("/alert")
def create_alert(alert: Alert):
    start = time.perf_counter()
    _ensure_csv()
    with open(ALERTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow([datetime.utcnow().isoformat(), alert.event_id, alert.entity_id,
                                alert.anomaly_score, alert.psi_value, alert.timestamp])
    return {"status": "alerted", "latency_ms": round((time.perf_counter() - start) * 1000, 3)}


# Returns the most recent alerts for quick inspection / debugging. In production
# this would be backed by a queryable store (Elasticsearch, a DB) with paging.
@app.get("/alerts")
def list_alerts():
    _ensure_csv()
    with open(ALERTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return {"count": len(rows[-50:]), "alerts": rows[-50:]}


# Liveness probe for orchestrators / load balancers. In production this would
# also check downstream dependencies (queue, notifier) before reporting healthy.
@app.get("/health")
def health():
    return {"status": "ok", "service": "alert_service"}
