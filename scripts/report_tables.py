"""Render markdown tables from reports/results.json (+ benchmark.json) so the report never needs hand-typed numbers.

    PYTHONPATH=. python scripts/report_tables.py   ->  reports/tables.md
"""
from __future__ import annotations

import json

from src import config
from src.registry import DISEASES, get_spec

R = json.loads((config.REPORTS_DIR / "results.json").read_text())
B_PATH = config.REPORTS_DIR / "benchmark.json"
B = json.loads(B_PATH.read_text()) if B_PATH.exists() else None
D = R["diseases"]
NAMES = {k: DISEASES[k].name for k in DISEASES}
out: list[str] = []


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ms(m):
    return f"{m['mean']:.3f} ± {m['std']:.3f}"


def table(title, header, rows):
    out.append(f"### {title}\n")
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    out.append("")


# 1 data ------------------------------------------------------------------------------------
rows = []
for k, d in D.items():
    x = d["data"]
    miss = ", ".join(f"{a}: {b}" for a, b in x["missing_after_sentinel_conversion"].items()) or "none"
    rows.append([NAMES[k], x["raw_rows"], x["duplicates_removed"], x["n_samples"], x["n_features"],
                 f"{x['class_counts']['positive']} / {x['class_counts']['negative']} ({pct(x['positive_rate'])})",
                 f"{x['n_train']} / {x['n_test']}", miss])
table("Datasets", ["Module", "Raw rows", "Duplicates removed", "Rows used", "Features",
                   "Positive / negative", "Train / test", "Missing after sentinel conversion"], rows)

# 2 baseline CV -----------------------------------------------------------------------------
for k, d in D.items():
    rows = []
    for n, v in d["cv_baseline"].items():
        rows.append([n, pct(v["accuracy"]["mean"]), ms(v["precision"]), ms(v["recall"]), ms(v["f1"]), ms(v["roc_auc"])])
    table(f"Baseline CV, default parameters: {NAMES[k]} (5-fold x 3 repeats on {d['data']['n_train']} training rows)",
          ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], rows)

# 3 tuning ---------------------------------------------------------------------------------
rows = []
for k, d in D.items():
    for n, t in d["tuning"].items():
        rows.append([NAMES[k], n, ", ".join(f"{a}={b}" for a, b in t["best_params"].items()),
                     f"{t['best_cv_roc_auc']:.4f}", t["n_candidates"], f"{t['wall_s']:.1f}"])
table("Grid search: best parameters", ["Module", "Model", "Best parameters", "CV ROC-AUC", "Combinations", "Search time (s)"], rows)

# 4 selection ------------------------------------------------------------------------------
for k, d in D.items():
    rows = []
    for n, v in sorted(d["selection"].items(), key=lambda kv: -kv[1]["roc_auc"]["mean"]):
        rows.append([n + (" (selected)" if n == d["selected_model"] else ""), ms(v["roc_auc"]), ms(v["f1"]), ms(v["recall"])])
    table(f"Selection round on new CV folds: {NAMES[k]}", ["Candidate", "ROC-AUC", "F1", "Recall"], rows)

# 5 hold-out -------------------------------------------------------------------------------
rows = []
for k, d in D.items():
    t, ci = d["test"], d["test_ci95"]
    c = t["confusion"]
    f = lambda m: f"{pct(t[m])} ({pct(ci[m][0])} to {pct(ci[m][1])})"
    rows.append([NAMES[k], d["selected_model"], t["n"], f("accuracy"), f("precision"), f("recall"), pct(t["specificity"]),
                 f("f1"), f"{t['roc_auc']:.3f} ({ci['roc_auc'][0]:.3f} to {ci['roc_auc'][1]:.3f})",
                 f"{t['mcc']:.3f}", f"{t['brier']:.3f}", f"TN {c['tn']}, FP {c['fp']}, FN {c['fn']}, TP {c['tp']}"])
table("Hold-out results at the 0.5 decision line (95% bootstrap intervals)",
      ["Module", "Selected model", "n", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "MCC", "Brier", "Confusion"], rows)

rows = []
for k, d in D.items():
    for n, v in d["test_reference_all_candidates"].items():
        rows.append([NAMES[k], n + (" (selected)" if n == d["selected_model"] else ""), pct(v["accuracy"]),
                     pct(v["recall"]), f"{v['f1']:.3f}", f"{v['roc_auc']:.3f}"])
table("Reference only: every candidate on the hold-out (not used for any decision)",
      ["Module", "Candidate", "Accuracy", "Recall", "F1", "ROC-AUC"], rows)

# 6 screening ------------------------------------------------------------------------------
rows = []
for k, d in D.items():
    s = d["screening"]
    rows.append([NAMES[k], f"{s['threshold']:.3f}", pct(s["oof_train"]["recall"]), pct(s["test"]["recall"]),
                 pct(s["test"]["specificity"]), pct(s["test"]["precision"]), pct(s["test_flag_rate"]),
                 pct(d["test"]["recall"]), pct(d["test"]["specificity"])])
table("Screening line versus the 0.5 decision line (hold-out)",
      ["Module", "Screening threshold", "Out-of-fold recall (train)", "Hold-out recall", "Hold-out specificity",
       "Hold-out precision", "Share flagged", "Recall at 0.5", "Specificity at 0.5"], rows)

# 7 stability ------------------------------------------------------------------------------
rows = []
for k, d in D.items():
    s = d["stability"]
    cell = lambda m, p=True: (f"{pct(s[m]['mean'])} ± {pct(s[m]['std'])} ({pct(s[m]['min'])} to {pct(s[m]['max'])})"
                              if p else f"{s[m]['mean']:.3f} ± {s[m]['std']:.3f} ({s[m]['min']:.3f} to {s[m]['max']:.3f})")
    rows.append([NAMES[k], len(s["rows"]), cell("accuracy"), cell("recall"), cell("f1"), cell("roc_auc", False), pct(d["test"]["accuracy"])])
table("Stability over 10 random 80/20 splits: mean ± SD (min to max)",
      ["Module", "Splits", "Accuracy", "Recall", "F1", "ROC-AUC", "Seed-42 accuracy (reported split)"], rows)

# 8 ablations ------------------------------------------------------------------------------
rows = []
for k, d in D.items():
    for variant, models in d["ablations"].items():
        for n, v in models.items():
            rows.append([NAMES[k], variant, n, ms(v["roc_auc"]), ms(v["f1"]), ms(v["recall"])])
table("Preprocessing ablations (5-fold x 3 CV on the training split)",
      ["Module", "Variant", "Model", "ROC-AUC", "F1", "Recall"], rows)

# 9 importance -----------------------------------------------------------------------------
for k, d in D.items():
    spec = get_spec(k)
    rows = [[i + 1, spec.feature(f["feature"]).label, f"{f['mean_auc_drop']:.3f} ± {f['std']:.3f}"]
            for i, f in enumerate(d["permutation_importance"][:5])]
    table(f"Permutation importance, top 5: {NAMES[k]}", ["Rank", "Feature", "Drop in ROC-AUC when shuffled"], rows)

# 10 performance ---------------------------------------------------------------------------
rows = []
for k, d in D.items():
    l = d["latency_model_only"]
    rows.append([NAMES[k], d["selected_model"], f"{d['final_fit_time_s']:.2f}", f"{d['total_wall_s']:.0f}",
                 f"{d['model_file_kb']:.0f}", f"{l['single_ms']['p50']:.2f}", f"{l['single_ms']['p95']:.2f}",
                 f"{l['single_ms']['p99']:.2f}", f"{l['batch']['rows_per_s']:.0f}"])
table("Model-level cost (in-process, one row per call)",
      ["Module", "Selected model", "Final fit (s)", "Whole protocol (s)", "Model file (KB)", "Predict p50 (ms)",
       "p95 (ms)", "p99 (ms)", "Batch throughput (rows/s)"], rows)

if B:
    rows = []
    for k, v in B["sequential"].items():
        for lab, s in v.items():
            rows.append([NAMES[k], lab.replace("_", " "), s["n"], f"{s['p50_ms']:.1f}", f"{s['p95_ms']:.1f}",
                         f"{s['p99_ms']:.1f}", f"{s['max_ms']:.1f}", s["http_errors"]])
    table("HTTP latency, one request at a time (includes TCP connect and SQLite write)",
          ["Module", "Mode", "Requests", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Max (ms)", "HTTP errors"], rows)
    rows = [[w, v["requests"], f"{v['throughput_rps']:.1f}", f"{v['p50_ms']:.1f}", f"{v['p95_ms']:.1f}", f"{v['p99_ms']:.1f}", v["http_errors"],
             f"{v['server_cpu_s']:.1f}"] for w, v in B["concurrent"].items()]
    table("Concurrent load, mixed modules, explanation off", ["Client threads", "Requests", "Throughput (req/s)", "p50 (ms)",
                                                              "p95 (ms)", "p99 (ms)", "HTTP errors", "Server CPU seconds"], rows)
    rows = []
    for k, v in B["batch"].items():
        for n, s in v.items():
            rows.append([NAMES[k], n, f"{s['total_ms']:.0f}", f"{s['rows_per_s']:.0f}", s["status"]])
    table("Batch endpoint", ["Module", "Rows per request", "Total (ms)", "Rows/s", "Status"], rows)
    out.append(f"Cold start (process launch to healthy): {B['cold_start_s']:.2f} s; model load: {B['model_load_s_all_modules']:.2f} s; "
               f"server memory idle {B['server_rss_idle_mb']:.0f} MB, peak {B['server_rss_peak_mb']:.0f} MB; "
               f"rows written to SQLite during the benchmark: {B['database_rows_written']}.\n")

env = R["meta"]["environment"]
out.append(f"Environment: {env['cpu_model']}, {env['cpu_logical_cores']} logical cores, {env['ram_total_gb']} GB RAM, {env['platform']}, "
           f"Python {env['python']}, scikit-learn {env['scikit_learn']}, NumPy {env['numpy']}, pandas {env['pandas']}. "
           f"Total training wall time {R['meta']['total_training_wall_s']:.0f} s; training-process peak RSS {R['meta']['peak_rss_mb_main_process']:.0f} MB.\n")
(config.REPORTS_DIR / "tables.md").write_text("\n".join(out))
print("\n".join(out))
