"""Per-customer adaptive similarity threshold stored in SQLite."""
from __future__ import annotations

import os
import sqlite3

_DB_PATH = ".tokenai_cache/thresholds.db"
_DEFAULT_THRESHOLD = 0.85
_THRESHOLD_MIN = 0.70
_THRESHOLD_MAX = 0.98

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS customer_thresholds (
    customer_id  TEXT PRIMARY KEY,
    threshold    REAL,
    total_hits   INTEGER,
    correct_hits INTEGER
)
"""


def _connect() -> sqlite3.Connection:
    """Open the DB, creating the directory and table if they don't exist."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def get(customer_id: str) -> float:
    """Return the stored threshold for *customer_id*, or 0.85 for new customers."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT threshold FROM customer_thresholds WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
    return row[0] if row else _DEFAULT_THRESHOLD


def update(customer_id: str, was_correct: bool) -> float:
    """Apply the adaptive update rule and return the new threshold.

    Correct hits loosen the threshold by 0.005 (more cache hits).
    Wrong hits tighten it by 0.020 (fewer false positives), 4× faster.
    Clamped to [0.70, 0.98].
    """
    current = get(customer_id)
    if was_correct:
        new_threshold = current - 0.005
    else:
        new_threshold = current + 0.020
    new_threshold = max(_THRESHOLD_MIN, min(_THRESHOLD_MAX, new_threshold))

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO customer_thresholds (customer_id, threshold, total_hits, correct_hits)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(customer_id) DO UPDATE SET
                threshold    = excluded.threshold,
                total_hits   = total_hits + 1,
                correct_hits = correct_hits + excluded.correct_hits
            """,
            (customer_id, new_threshold, 1 if was_correct else 0),
        )
    return new_threshold


def get_stats(customer_id: str) -> dict:
    """Return threshold statistics for *customer_id*.

    Returns ``{"threshold", "total_hits", "correct_hits", "accuracy"}``.
    New customers get default values with zero hit counts.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT threshold, total_hits, correct_hits FROM customer_thresholds WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()

    if row is None:
        return {
            "threshold": _DEFAULT_THRESHOLD,
            "total_hits": 0,
            "correct_hits": 0,
            "accuracy": 0.0,
        }

    threshold, total_hits, correct_hits = row
    accuracy = correct_hits / total_hits if total_hits > 0 else 0.0
    return {
        "threshold": threshold,
        "total_hits": total_hits,
        "correct_hits": correct_hits,
        "accuracy": accuracy,
    }
