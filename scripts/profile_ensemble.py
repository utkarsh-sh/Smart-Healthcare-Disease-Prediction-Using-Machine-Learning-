"""Per-member prediction latency of the diabetes soft-voting ensemble (bottleneck analysis).

    PYTHONPATH=. python scripts/profile_ensemble.py
"""
import time, joblib, numpy as np, pandas as pd
from src import config
m = joblib.load(config.MODELS_DIR / "diabetes_model.joblib")
X = pd.read_csv(config.MODELS_DIR / "diabetes_holdout.csv").drop(columns="y")
rows = [X.iloc[[i % len(X)]] for i in range(150)]
def bench(est, label):
    for r in rows[:5]: est.predict_proba(r)
    t = []
    for r in rows:
        s = time.perf_counter(); est.predict_proba(r); t.append((time.perf_counter() - s) * 1000)
    print(f"{label:<26} p50 {np.percentile(t,50):6.2f} ms   p95 {np.percentile(t,95):6.2f} ms")
bench(m, "ensemble (whole)")
for name, est in m.named_estimators_.items():
    extra = ""
    clf = est.named_steps["clf"]
    if hasattr(clf, "n_estimators"): extra = f"  [{clf.n_estimators} trees]"
    bench(est, name + extra)
