"""Reproducibility check: compare two results.json files and report every differing value.

    PYTHONPATH=. python scripts/compare_runs.py reports/results.json /path/to/second_run/results.json

Keys that only measure time or memory (wall_s, fit_time_s, latency summaries, throughput, timestamps, RSS) are
expected to differ between runs and are reported separately from metrics, which must be identical.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

TIMING_KEYS = {"wall_s", "fit_time_s", "final_fit_time_s", "total_wall_s", "total_training_wall_s", "generated_at",
               "peak_rss_mb_main_process"}
TIMING_SECTIONS = ("latency_model_only",)          # latency and throughput summaries live under this key


def is_timing(path: str) -> bool:
    return path.split("/")[-1] in TIMING_KEYS or any(s in path for s in TIMING_SECTIONS)


def leaves_that_differ(a, b, path=""):
    if isinstance(a, dict):
        for k in a:
            yield from leaves_that_differ(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"length differs at {path}"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from leaves_that_differ(x, y, f"{path}[{i}]")
    elif a != b:
        yield path


def main(p1: str, p2: str) -> int:
    a, b = json.load(open(p1)), json.load(open(p2))
    diffs = list(leaves_that_differ(a, b))
    timing = [d for d in diffs if is_timing(d)]
    metric = [d for d in diffs if not is_timing(d)]
    print(f"compared {p1} with {p2}")
    print(f"differing values: {len(diffs)} (timing or memory: {len(timing)}, metrics or parameters: {len(metric)})")
    print("timing keys that differ:", dict(Counter(d.split('/')[-1] for d in timing)))
    for d in metric:
        print("METRIC DIFFERS:", d)
    print("RESULT:", "metrics identical" if not metric else "METRICS DIFFER")
    return 1 if metric else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
