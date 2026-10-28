"""SQLite audit log of predictions (database layer of the prototype).

Schema (one table, one index):

    predictions(id PK, created_at, disease, inputs_json, probability, prediction,
                screening_flag, model_name, model_version, latency_ms)

No patient identifiers are collected: only the feature values the user typed and the model output.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT    NOT NULL,
    disease        TEXT    NOT NULL,
    inputs_json    TEXT    NOT NULL,
    probability    REAL    NOT NULL CHECK (probability BETWEEN 0 AND 1),
    prediction     INTEGER NOT NULL CHECK (prediction IN (0, 1)),
    screening_flag INTEGER NOT NULL CHECK (screening_flag IN (0, 1)),
    model_name     TEXT    NOT NULL,
    model_version  TEXT    NOT NULL,
    latency_ms     REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_disease_time ON predictions (disease, created_at);
"""


class PredictionStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        """Yield a connection that commits/rolls back AND is always closed (sqlite3's own
        context manager only commits, which leaked one open connection per request)."""
        con = sqlite3.connect(self.path, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def add(self, result: dict, inputs: dict) -> int:
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO predictions (created_at, disease, inputs_json, probability, prediction, "
                "screening_flag, model_name, model_version, latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), result["disease"],
                 json.dumps(inputs, sort_keys=True), result["probability"], result["prediction"],
                 int(result["screening_flag"]), result["model"]["name"], result["model"]["version"],
                 result["latency_ms"]))
            return int(cur.lastrowid)

    def recent(self, limit: int = 20, disease: str | None = None) -> list[dict]:
        sql = "SELECT * FROM predictions"
        args: list = []
        if disease:
            sql += " WHERE disease = ?"
            args.append(disease)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._connect() as con:
            rows = con.execute(sql, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["inputs"] = json.loads(d.pop("inputs_json"))
            out.append(d)
        return out

    def counts(self) -> dict:
        with self._connect() as con:
            rows = con.execute("SELECT disease, COUNT(*) AS n FROM predictions GROUP BY disease").fetchall()
        return {r["disease"]: r["n"] for r in rows}

    def clear(self) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM predictions")
