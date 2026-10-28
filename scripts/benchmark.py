"""System-level performance benchmark of the REST API over real HTTP.

The API runs in a *separate* process (Werkzeug threaded dev server); this script is the load generator.
Every request opens a new TCP connection (the dev server speaks HTTP/1.0), so latencies include connect time.
Predictions are written to a temporary SQLite database exactly as in normal operation.

    PYTHONPATH=. python scripts/benchmark.py            ->  reports/benchmark.json
"""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import psutil

from src import config
from src.predictor import DiseasePredictor
from src.registry import DISEASES, get_spec

PORT = 5077
N_SEQUENTIAL = 300
CONCURRENCY_LEVELS = (1, 4, 8)
N_CONCURRENT = 400


def holdout_payloads(key: str, n: int) -> list[dict]:
    spec = get_spec(key)
    hold = pd.read_csv(config.MODELS_DIR / f"{key}_holdout.csv")[spec.feature_names]
    rows = []
    for i in range(n):
        r = hold.iloc[i % len(hold)]
        p = {}
        for f in spec.features:
            v = float(r[f.name])
            if v in spec.sentinels.get(f.name, []):
                p[f.name] = None
            else:
                p[f.name] = v if f.kind == "numeric" else int(v)
        rows.append(p)
    return rows


def post(path: str, body: dict) -> tuple[int, float]:
    t = time.perf_counter()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp.status, (time.perf_counter() - t) * 1000


def pct(a, q):
    return float(np.percentile(a, q))


def summarise(ms: list[float]) -> dict:
    return {"n": len(ms), "mean_ms": float(np.mean(ms)), "p50_ms": pct(ms, 50), "p95_ms": pct(ms, 95),
            "p99_ms": pct(ms, 99), "max_ms": float(np.max(ms))}


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="bt60_bench_")
    code = ("from app.api import create_app; "
            f"create_app({{'DB_PATH': r'{tmp}/bench.db'}}).run(host='127.0.0.1', port={PORT}, threaded=True, debug=False)")
    env = dict(os.environ, PYTHONPATH=str(config.ROOT), PYTHONWARNINGS="ignore")
    t_launch = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=config.ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ps = psutil.Process(proc.pid)
    try:
        # ---- cold start --------------------------------------------------------------------
        while True:
            try:
                c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=2)
                c.request("GET", "/api/health")
                if c.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.1)
        cold_start_s = time.perf_counter() - t_launch
        rss_idle = ps.memory_info().rss / 1024 ** 2

        out = {"environment": {"python": sys.version.split()[0], "cpu_logical_cores": os.cpu_count(),
                               "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1),
                               "server": "Werkzeug threaded dev server (separate process), client threads in the benchmark process"},
               "cold_start_s": cold_start_s, "server_rss_idle_mb": rss_idle}

        # model load time measured in-process
        t = time.perf_counter()
        DiseasePredictor()
        out["model_load_s_all_modules"] = time.perf_counter() - t

        # ---- sequential latency -------------------------------------------------------------
        seq = {}
        peak_rss = rss_idle
        for key in DISEASES:
            rows = holdout_payloads(key, N_SEQUENTIAL + 20)
            for r in rows[:20]:                                  # warm-up
                post(f"/api/predict/{key}", r)
            res = {}
            for label, qs in (("with_explanation", ""), ("without_explanation", "?explain=false")):
                ms, bad = [], 0
                for r in rows[20:]:
                    status, dt = post(f"/api/predict/{key}{qs}", r)
                    bad += status != 200
                    ms.append(dt)
                res[label] = {**summarise(ms), "http_errors": bad}
            seq[key] = res
            peak_rss = max(peak_rss, ps.memory_info().rss / 1024 ** 2)
        out["sequential"] = seq

        # ---- concurrency / throughput ---------------------------------------------------------
        keys = list(DISEASES)
        work = [(keys[i % 3], holdout_payloads(keys[i % 3], 1 + i % 40)[-1]) for i in range(N_CONCURRENT)]
        conc = {}
        for workers in CONCURRENCY_LEVELS:
            cpu0 = ps.cpu_times()
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda kw: post(f"/api/predict/{kw[0]}?explain=false", kw[1]), work))
            wall = time.perf_counter() - t0
            cpu1 = ps.cpu_times()
            ms = [r[1] for r in results]
            conc[str(workers)] = {**summarise(ms), "requests": len(results), "wall_s": wall,
                                  "throughput_rps": len(results) / wall,
                                  "http_errors": int(sum(r[0] != 200 for r in results)),
                                  "server_cpu_s": (cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)}
            peak_rss = max(peak_rss, ps.memory_info().rss / 1024 ** 2)
        out["concurrent"] = conc

        # ---- batch endpoint ---------------------------------------------------------------------
        batch = {}
        for key in DISEASES:
            batch[key] = {}
            for n in (100, 500):
                rows = holdout_payloads(key, n)
                status, dt = post(f"/api/predict/{key}/batch", {"rows": rows})
                batch[key][str(n)] = {"status": status, "total_ms": dt, "rows_per_s": n / (dt / 1000)}
        out["batch"] = batch
        out["server_rss_peak_mb"] = max(peak_rss, ps.memory_info().rss / 1024 ** 2)
        out["database_rows_written"] = sum(
            1 for _ in [0]) and __import__("sqlite3").connect(f"{tmp}/bench.db").execute(
            "SELECT COUNT(*) FROM predictions").fetchone()[0]

        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.REPORTS_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("cold_start_s", "server_rss_idle_mb", "server_rss_peak_mb",
                                                "model_load_s_all_modules", "database_rows_written")}, indent=2))
        for key, v in seq.items():
            s = v["with_explanation"]
            print(f"{key:9s} sequential p50 {s['p50_ms']:.1f} ms  p95 {s['p95_ms']:.1f} ms  p99 {s['p99_ms']:.1f} ms  errors {s['http_errors']}")
        for w, v in conc.items():
            print(f"workers {w}: {v['throughput_rps']:.1f} req/s  p95 {v['p95_ms']:.1f} ms  errors {v['http_errors']}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
