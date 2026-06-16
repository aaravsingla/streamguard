# Author: Aarav Singla

"""
PSI-based concept drift detection.

WHAT IS PSI (Population Stability Index)?
-----------------------------------------
PSI measures how much a feature's *distribution* has shifted between a
reference window (the data the model was trained on) and a new, incoming
window of live data. We bucket both distributions into the same bins and
compare the proportion of samples that fall into each bucket.

    PSI = sum( (actual_prop - expected_prop) * ln(actual_prop / expected_prop) )

- expected_prop : fraction of the *training* data in a bin
- actual_prop   : fraction of the *incoming* batch in that same bin

Intuition: if the live data lands in the same bins, in the same
proportions, as the training data, every term is ~0 and PSI ~= 0. As the
live distribution drifts away, the proportions diverge and PSI grows.

WHY THRESHOLD = 0.1?
--------------------
This is the widely used industry rule of thumb (originating in credit
scoring / model monitoring):
    PSI < 0.1   -> no significant shift, model still trustworthy
    0.1 <= PSI < 0.25 -> moderate shift, investigate / consider retraining
    PSI >= 0.25 -> major shift, distribution has clearly moved
We trigger at PSI > 0.1 so drift is flagged early, before the model's
inputs have moved far enough to seriously degrade predictions.
"""

import csv
import os
from datetime import datetime

import numpy as np

# Small constant used to replace zero proportions. ln(0) is undefined and a
# zero in a denominator blows the PSI term up to infinity, so we floor every
# proportion at EPSILON before applying the formula.
EPSILON = 1e-6

# Default drift threshold (see module docstring for the rationale).
DEFAULT_THRESHOLD = 0.1

# logs/drift_events.csv lives at the repo root (this file is in streaming/).
DEFAULT_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "logs", "drift_events.csv")
)

CSV_HEADER = ["timestamp", "feature_name", "psi_value", "triggered_retrain"]


class DriftDetector:
    """Detect concept drift on a set of numeric features using PSI.

    The detector is initialised with the *training* feature distributions
    (mean, std and histogram bin edges + expected proportions per bin). For
    each incoming batch it re-bins the live data with the exact same edges
    and computes PSI per feature.
    """

    def __init__(self, feature_distributions, threshold=DEFAULT_THRESHOLD,
                 log_path=DEFAULT_LOG_PATH):
        """
        Parameters
        ----------
        feature_distributions : dict
            Maps feature_name -> dict with keys:
                "mean"           : float, training mean (kept for reference)
                "std"            : float, training std  (kept for reference)
                "bin_edges"      : 1D array of histogram edges (len = n_bins+1)
                "expected_props" : 1D array of training proportions per bin
                                   (len = n_bins, sums to 1.0)
        threshold : float
            PSI value above which drift is flagged (default 0.1).
        log_path : str
            Where drift events are appended as CSV rows.
        """
        self.feature_distributions = feature_distributions
        self.feature_names = list(feature_distributions.keys())
        self.threshold = threshold
        self.log_path = log_path
        self._ensure_log_file()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_training_data(cls, X, feature_names=None, n_bins=10,
                           threshold=DEFAULT_THRESHOLD, log_path=DEFAULT_LOG_PATH):
        """Build a detector directly from the training feature matrix.

        Bins are chosen on the training data and then *frozen*; the same
        edges are reused for every incoming batch so the comparison is
        always apples-to-apples.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = ["feature_{}".format(i) for i in range(n_features)]

        feature_distributions = {}
        for i, name in enumerate(feature_names):
            col = X[:, i]
            # Histogram edges define the buckets. We use the data range so the
            # training data is fully covered.
            counts, bin_edges = np.histogram(col, bins=n_bins)
            expected_props = counts / counts.sum()
            feature_distributions[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "bin_edges": bin_edges,
                "expected_props": expected_props,
            }

        return cls(feature_distributions, threshold=threshold, log_path=log_path)

    # ------------------------------------------------------------------ #
    # Core PSI computation
    # ------------------------------------------------------------------ #
    def compute_psi(self, incoming_batch):
        """Compute PSI for every known feature against an incoming batch.

        Parameters
        ----------
        incoming_batch : array-like, shape (n_samples, n_features)
            The live feature rows. Column order must match the order of
            ``self.feature_names``.

        Returns
        -------
        dict with keys:
            "psi"       : {feature_name: psi_value (float)}
            "triggered" : bool, True if ANY feature's PSI > threshold
        Features that cannot be evaluated (missing column, empty batch)
        get a psi_value of None and never trigger.
        """
        timestamp = datetime.utcnow().isoformat()

        # --- Edge case: empty / None batch -> nothing to score ---------- #
        batch = np.asarray(incoming_batch, dtype=float) if incoming_batch is not None \
            else np.empty((0, 0))
        if batch.size == 0:
            empty = {name: None for name in self.feature_names}
            return {"psi": empty, "triggered": False}

        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)

        n_cols = batch.shape[1]
        psi_values = {}
        triggered = False

        for idx, name in enumerate(self.feature_names):
            # --- Edge case: feature missing from this batch ------------- #
            if idx >= n_cols:
                psi_values[name] = None
                self._log_event(timestamp, name, None, False)
                continue

            dist = self.feature_distributions[name]
            bin_edges = dist["bin_edges"]
            expected_props = dist["expected_props"]

            # Re-bin the incoming column with the FROZEN training edges so the
            # two distributions are directly comparable.
            actual_counts, _ = np.histogram(batch[:, idx], bins=bin_edges)
            actual_total = actual_counts.sum()
            if actual_total == 0:
                # No incoming sample fell inside the training range at all.
                psi_values[name] = None
                self._log_event(timestamp, name, None, False)
                continue

            actual_props = actual_counts / actual_total

            psi_value = self._psi(expected_props, actual_props)
            feature_triggered = psi_value > self.threshold
            triggered = triggered or feature_triggered

            psi_values[name] = psi_value
            self._log_event(timestamp, name, psi_value, feature_triggered)

        return {"psi": psi_values, "triggered": triggered}

    @staticmethod
    def _psi(expected_props, actual_props):
        """Apply the PSI formula bin-by-bin.

        Zero proportions are floored at EPSILON because the formula relies on
        ln(actual/expected): a literal zero would make the log term either
        -inf, +inf or NaN. Flooring keeps each term finite while still
        registering the divergence.
        """
        expected = np.clip(expected_props, EPSILON, None)
        actual = np.clip(actual_props, EPSILON, None)
        psi_terms = (actual - expected) * np.log(actual / expected)
        return float(np.sum(psi_terms))

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _ensure_log_file(self):
        """Create logs/ and the CSV header on first use."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

    def _log_event(self, timestamp, feature_name, psi_value, triggered_retrain):
        """Append a single drift-evaluation row to the CSV log."""
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                timestamp,
                feature_name,
                "" if psi_value is None else round(psi_value, 6),
                triggered_retrain,
            ])
