"""
Lightweight SQLite persistence for real usage history.

Every dashboard number (total analyses, change detections run, VQA
queries, average confidence) is computed from rows actually written here
when a request completes — never hardcoded/fabricated. On a fresh
install the dashboard correctly starts at zero.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "satquery.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    query TEXT,
    answer TEXT,
    confidence REAL,
    processing_time_s REAL,
    created_at REAL NOT NULL,
    extra_json TEXT
);
CREATE TABLE IF NOT EXISTS batch_tasks (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT,
    result_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.executescript(SCHEMA)


def log_analysis(task_type: str, query: str, answer: str, confidence: float,
                  processing_time_s: float, extra: dict) -> str:
    aid = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO analyses (id, task_type, query, answer, confidence, "
            "processing_time_s, created_at, extra_json) VALUES (?,?,?,?,?,?,?,?)",
            (aid, task_type, query, answer, confidence, processing_time_s,
             time.time(), json.dumps(extra, default=str)),
        )
    return aid


def dashboard_stats() -> dict:
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) n FROM analyses").fetchone()["n"]
        changes = c.execute(
            "SELECT COUNT(*) n FROM analyses WHERE task_type IN "
            "('detect_change','change_trend')"
        ).fetchone()["n"]
        vqa = c.execute(
            "SELECT COUNT(*) n FROM analyses WHERE task_type='describe_image'"
        ).fetchone()["n"]
        avg_conf_row = c.execute("SELECT AVG(confidence) a FROM analyses").fetchone()
        avg_conf = avg_conf_row["a"] or 0.0
        recent = c.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
    return {
        "total_analyses": total,
        "change_detections": changes,
        "vqa_queries": vqa,
        "avg_accuracy_pct": round(avg_conf * 100, 1),
        "recent": [dict(r) for r in recent],
    }


def history(limit: int = 100) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------- batch tasks ----------------

def create_batch_task(task_type: str, payload: dict) -> str:
    tid = str(uuid.uuid4())
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO batch_tasks (id, task_type, status, progress, payload_json, "
            "result_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (tid, task_type, "queued", 0, json.dumps(payload), None, now, now),
        )
    return tid


def update_batch_task(task_id: str, status: str, progress: int, result: dict | None = None):
    with _conn() as c:
        c.execute(
            "UPDATE batch_tasks SET status=?, progress=?, result_json=?, updated_at=? WHERE id=?",
            (status, progress, json.dumps(result, default=str) if result else None,
             time.time(), task_id),
        )


def get_batch_task(task_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM batch_tasks WHERE id=?", (task_id,)).fetchone()
    return dict(row) if row else None


def list_batch_tasks() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM batch_tasks ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]
