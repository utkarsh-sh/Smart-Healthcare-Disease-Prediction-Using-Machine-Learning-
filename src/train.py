"""End-to-end training and evaluation for every disease module.

Usage (from project root):
    PYTHONPATH=. python -m src.train                # all diseases (full protocol)
    PYTHONPATH=. python -m src.train --only heart   # one disease
    PYTHONPATH=. python -m src.train --fast         # tiny grids, for debugging only

Protocol (identical for every disease, fixed before looking at results)
  1. Stratified 80/20 split (seed 42). The 20 % hold-out set is used ONCE, at the very end.
  2. Baseline round : 5-fold x 3-repeat stratified CV on the training part, default hyper-parameters.
  3. Tuning round   : GridSearchCV (5-fold, scoring = ROC-AUC) for every candidate model.
  4. Selection round: tuned models + soft-voting ensemble compared with a *different* 5x3 CV
                      (seed 7); highest mean ROC-AUC wins, ties (<= 0.002) go to the simpler model.
  5. Final model fitted on the training part, evaluated once on the hold-out set
     (+ bootstrap 95 % CIs, screening threshold, permutation importance, latency).
  6. Stability check: 10 further random splits with the selected configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

from . import config
from .metrics import bootstrap_ci, classification_metrics, roc_points, screening_threshold
from .modeling import (GRIDS, SIMPLICITY_ORDER, baseline_dummy, build_ensemble, build_pipeline, model_zoo)
from .registry import DISEASES, Feature, get_spec, load_breast
from .transformers import SentinelToNaN

warnings.filterwarnings("ignore")
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def cv_summary(est, X, y, cv) -> dict:
    t = time.perf_counter()
    res = cross_validate(est, X, y, cv=cv, scoring=SCORING, n_jobs=2, error_score="raise")
    out = {m: {"mean": float(np.mean(res[f"test_{m}"])), "std": float(np.std(res[f"test_{m}"], ddof=1))}
           for m in SCORING}
    out["fit_time_s"] = float(np.mean(res["fit_time"]))
    out["n_fits"] = int(len(res["fit_time"]))
    out["wall_s"] = float(time.perf_counter() - t)
    return out


def choose_model(selection: dict, tol: float = 0.002) -> str:
    best = max(v["roc_auc"]["mean"] for v in selection.values())
    contenders = [n for n, v in selection.items() if best - v["roc_auc"]["mean"] <= tol + 1e-9]  # eps: float-safe tie test
    return min(contenders, key=SIMPLICITY_ORDER.index)


def environment_info() -> dict:
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "cpu_model": cpu, "cpu_logical_cores": os.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1)}


def numeric_train_stats(spec, X_train: pd.DataFrame) -> dict:
    """Per-feature statistics from the training split (after sentinel->NaN) for OOD warnings / occlusion baselines."""
    Xc = SentinelToNaN(spec.sentinels).fit_transform(X_train)
    stats = {}
    for f in spec.features:
        col = Xc[f.name].dropna()
        if f.kind in ("numeric", "integer"):
            stats[f.name] = {"p1": float(col.quantile(0.01)), "p99": float(col.quantile(0.99)),
                             "min": float(col.min()), "max": float(col.max()),
                             "median": float(col.median()), "mean": float(col.mean()), "std": float(col.std()),
                             "baseline": float(col.median())}
        else:
            vc = col.value_counts(normalize=True)
            stats[f.name] = {"baseline": float(col.mode().iloc[0]),
                             "distribution": {str(int(k)): float(v) for k, v in vc.items()}}
    return stats


def typical_examples(spec, final, X_test, y_test) -> dict:
    """One 'typical' true-positive and true-negative from the hold-out set (closest to class-median score)."""
    proba = final.predict_proba(X_test)[:, 1]
    out = {}
    for name, cls in (("positive", 1), ("negative", 0)):
        idx = np.where(y_test.to_numpy() == cls)[0]
        med = np.median(proba[idx])
        pick = idx[np.argmin(np.abs(proba[idx] - med))]
        row = SentinelToNaN(spec.sentinels).fit_transform(X_test.iloc[[pick]]).iloc[0]
        values = {}
        for f in spec.features:
            v = row[f.name]
            values[f.name] = None if pd.isna(v) else (int(v) if f.kind in ("integer", "binary", "category")
                                                       else round(float(v), 4))
        out[name] = {"values": values, "true_label": cls, "model_probability": float(proba[pick]),
                     "source": "held-out test row #%d" % int(X_test.index[pick])}
    return out


def latency_profile(final, X_test, n_calls: int = 300) -> dict:
    rows = [X_test.iloc[[i % len(X_test)]] for i in range(n_calls)]
    for r in rows[:5]:                              # warm-up
        final.predict_proba(r)
    times = []
    for r in rows:
        t = time.perf_counter()
        final.predict_proba(r)
        times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    t = time.perf_counter()
    final.predict_proba(X_test)
    batch_ms = (time.perf_counter() - t) * 1000
    return {"single_ms": {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                          "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                          "max": float(a.max()), "n_calls": n_calls},
            "batch": {"rows": int(len(X_test)), "total_ms": float(batch_ms),
                      "rows_per_s": float(len(X_test) / (batch_ms / 1000))}}


# --------------------------------------------------------------------------------------
# per-disease run
# --------------------------------------------------------------------------------------

def run_disease(key: str, fast: bool = False) -> dict:
    spec = get_spec(key)
    t_start = time.perf_counter()
    log(f"=== {spec.name} ===")
    X, y, info = spec.loader()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    missing = SentinelToNaN(spec.sentinels).fit_transform(X).isna().sum()
    data_report = {
        **info, "source": spec.source, "n_samples": int(len(X)), "n_features": int(X.shape[1]),
        "class_counts": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "positive_rate": float(y.mean()),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()), "test_positive_rate": float(y_test.mean()),
        "missing_after_sentinel_conversion": {k: int(v) for k, v in missing.items() if v > 0},
        "missing_pct": {k: round(100 * v / len(X), 1) for k, v in missing.items() if v > 0},
    }
    log(f"data: {data_report['n_samples']} rows, train {len(X_train)} / test {len(X_test)}, "
        f"positive rate {y.mean():.3f}, missing {data_report['missing_after_sentinel_conversion']}")

    repeats = 1 if fast else config.CV_REPEATS
    base_cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=repeats, random_state=config.RANDOM_STATE)

    # ---- 2. baseline CV with default hyper-parameters --------------------------------
    zoo = model_zoo(config.RANDOM_STATE)
    cv_baseline = {"Majority-class baseline": cv_summary(
        build_pipeline(spec.features, spec.sentinels, baseline_dummy()), X_train, y_train, base_cv)}
    for name, est in zoo.items():
        cv_baseline[name] = cv_summary(build_pipeline(spec.features, spec.sentinels, est), X_train, y_train, base_cv)
        log(f"baseline CV  {name:<22} AUC {cv_baseline[name]['roc_auc']['mean']:.4f}  F1 {cv_baseline[name]['f1']['mean']:.4f}")

    # ---- 3. hyper-parameter tuning ---------------------------------------------------
    tuning, tuned = {}, {}
    tune_cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    for name, est in zoo.items():
        grid = GRIDS[name]
        if fast:
            grid = {k: v[:2] for k, v in grid.items()}
        t = time.perf_counter()
        gs = GridSearchCV(build_pipeline(spec.features, spec.sentinels, est), grid, scoring="roc_auc",
                          cv=tune_cv, n_jobs=2, refit=True)
        gs.fit(X_train, y_train)
        tuned[name] = gs.best_estimator_
        tuning[name] = {"best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
                        "best_cv_roc_auc": float(gs.best_score_),
                        "n_candidates": int(len(gs.cv_results_["params"])),
                        "n_fits": int(len(gs.cv_results_["params"]) * config.CV_FOLDS),
                        "search_space": {k.replace("clf__", ""): [str(x) for x in v] for k, v in grid.items()},
                        "wall_s": float(time.perf_counter() - t)}
        log(f"tuned        {name:<22} CV-AUC {gs.best_score_:.4f}  {tuning[name]['best_params']}")

    # ---- 4. model selection (different folds; includes the ensemble) ------------------
    sel_cv = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=repeats, random_state=config.SELECTION_CV_SEED)
    candidates = dict(tuned)
    candidates["Soft-Voting Ensemble"] = build_ensemble(tuned)
    selection = {n: cv_summary(clone(m), X_train, y_train, sel_cv) for n, m in candidates.items()}
    selected = choose_model(selection)
    for n, v in selection.items():
        log(f"selection CV {n:<22} AUC {v['roc_auc']['mean']:.4f} ± {v['roc_auc']['std']:.4f}{'   <-- selected' if n == selected else ''}")

    # ---- 5. final fit + one-shot hold-out evaluation ---------------------------------
    final = clone(candidates[selected])
    t = time.perf_counter()
    final.fit(X_train, y_train)
    final_fit_s = time.perf_counter() - t
    proba = final.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, proba, 0.5)
    ci = bootstrap_ci(y_test, proba, 200 if fast else config.BOOTSTRAP_RESAMPLES, config.RANDOM_STATE)
    log(f"HOLD-OUT     {selected}: acc {test_metrics['accuracy']:.4f} rec {test_metrics['recall']:.4f} "
        f"f1 {test_metrics['f1']:.4f} auc {test_metrics['roc_auc']:.4f}")

    # reference only: every candidate on the hold-out set (NOT used for any decision)
    reference = {}
    for n, m in candidates.items():
        mm = clone(m).fit(X_train, y_train)
        reference[n] = classification_metrics(y_test, mm.predict_proba(X_test)[:, 1], 0.5)
        reference[n] = {k: reference[n][k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}

    # screening threshold chosen on out-of-fold training predictions only
    oof = cross_val_predict(clone(final), X_train, y_train,
                            cv=StratifiedKFold(5, shuffle=True, random_state=1),
                            method="predict_proba", n_jobs=2)[:, 1]
    thr = screening_threshold(y_train, oof, config.SCREENING_TARGET_RECALL)
    screening = {"target_recall": config.SCREENING_TARGET_RECALL, "threshold": thr,
                 "oof_train": classification_metrics(y_train, oof, thr),
                 "test": classification_metrics(y_test, proba, thr),
                 "test_flag_rate": float((proba >= thr).mean())}
    log(f"screening    thr {thr:.3f}: test recall {screening['test']['recall']:.3f}, "
        f"specificity {screening['test']['specificity']:.3f}")

    # ---- permutation importance -------------------------------------------------------
    pi = permutation_importance(final, X_test, y_test, scoring="roc_auc",
                                n_repeats=10 if fast else 30, random_state=config.RANDOM_STATE, n_jobs=1)
    importance = sorted(({"feature": f, "mean_auc_drop": float(m), "std": float(s)}
                         for f, m, s in zip(X.columns, pi.importances_mean, pi.importances_std)),
                        key=lambda d: -d["mean_auc_drop"])

    # ---- 6. stability over 10 other random splits ------------------------------------
    stab_rows = []
    for seed in config.STABILITY_SEEDS[: (3 if fast else None)]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
        p = clone(candidates[selected]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        m = classification_metrics(yte, p, 0.5)
        stab_rows.append({"seed": seed, **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
    stability = {"seeds": [r["seed"] for r in stab_rows], "rows": stab_rows}
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        v = np.array([r[k] for r in stab_rows])
        stability[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
    log(f"stability    acc {stability['accuracy']['mean']:.4f} ± {stability['accuracy']['std']:.4f} over {len(stab_rows)} splits")

    # ---- ablations ---------------------------------------------------------------------
    ablations = run_ablations(key, spec, X_train, y_train, base_cv, fast)

    # ---- persistence ---------------------------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / f"{key}_model.joblib"
    joblib.dump(final, model_path, compress=3)
    size_kb = model_path.stat().st_size / 1024
    latency = latency_profile(final, X_test)

    # deterministic hold-out artefacts used by the regression test (artefact integrity)
    holdout_path = config.MODELS_DIR / f"{key}_holdout.csv"
    pd.concat([X_test, y_test.rename("y")], axis=1).to_csv(holdout_path, index=False)

    train_stats = numeric_train_stats(spec, X_train)
    meta = {
        "key": key, "name": spec.name, "description": spec.short_description, "version": config.MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": selected,
        "selected_params": tuning.get(selected, {}).get("best_params"),
        "selection_rule": "highest mean ROC-AUC in 5x3 repeated CV on the training split; ties (<=0.002) -> simpler model",
        "positive_label": spec.positive_label, "negative_label": spec.negative_label,
        "features": [f.to_dict() for f in spec.features], "sentinels": spec.sentinels,
        "thresholds": {"default": 0.5, "screening": thr, "screening_target_recall": config.SCREENING_TARGET_RECALL},
        "train_stats": train_stats,
        "metrics": {"test": test_metrics, "test_ci95": ci, "screening_test": screening["test"]},
        "examples": typical_examples(spec, final, X_test, y_test),
        "source": spec.source, "population_note": spec.population_note,
        "random_state": config.RANDOM_STATE, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "environment": environment_info(),
    }
    (config.MODELS_DIR / f"{key}_meta.json").write_text(json.dumps(meta, indent=2))

    log(f"saved {model_path.name} ({size_kb:.0f} KB); total {time.perf_counter() - t_start:.1f}s")
    return {
        "data": data_report, "cv_baseline": cv_baseline, "tuning": tuning, "selection": selection,
        "selected_model": selected, "test": test_metrics, "test_ci95": ci,
        "test_reference_all_candidates": reference, "roc": roc_points(y_test, proba),
        "screening": screening, "permutation_importance": importance, "stability": stability,
        "ablations": ablations, "latency_model_only": latency,
        "final_fit_time_s": float(final_fit_s), "model_file_kb": float(size_kb),
        "total_wall_s": float(time.perf_counter() - t_start),
    }


def run_ablations(key, spec, X_train, y_train, cv, fast) -> dict:
    """Small, honest experiments that justify preprocessing decisions."""
    out = {}
    probes = {"Logistic Regression": model_zoo(config.RANDOM_STATE)["Logistic Regression"],
              "Random Forest": model_zoo(config.RANDOM_STATE)["Random Forest"]}
    if key == "diabetes":
        for variant, sent in (("zeros_treated_as_missing (ours)", spec.sentinels), ("zeros_kept_as_values", {})):
            out[variant] = {n: cv_summary(build_pipeline(spec.features, sent, clone(e)), X_train, y_train, cv)
                            for n, e in probes.items()}
            log(f"ablation diabetes/{variant}: " + ", ".join(
                f"{n} AUC {v['roc_auc']['mean']:.4f}" for n, v in out[variant].items()))
    elif key == "breast":
        X30, y30, _ = load_breast("all30")
        tr30, _, ytr30, _ = train_test_split(X30, y30, test_size=config.TEST_SIZE, stratify=y30,
                                             random_state=config.RANDOM_STATE)
        feats30 = tuple(Feature(c, c, "numeric") for c in X30.columns)
        out["mean10_features (ours)"] = {n: cv_summary(build_pipeline(spec.features, {}, clone(e)), X_train, y_train, cv)
                                         for n, e in probes.items()}
        out["all30_features"] = {n: cv_summary(build_pipeline(feats30, {}, clone(e)), tr30, ytr30, cv)
                                 for n, e in probes.items()}
        for variant, res in out.items():
            log(f"ablation breast/{variant}: " + ", ".join(f"{n} AUC {v['roc_auc']['mean']:.4f}" for n, v in res.items()))
    elif key == "heart":
        for variant, sent in (("invalid_codes_ca4_thal0_to_missing (ours)", spec.sentinels), ("invalid_codes_kept_as_values", {})):
            out[variant] = {n: cv_summary(build_pipeline(spec.features, sent, clone(e)), X_train, y_train, cv)
                            for n, e in probes.items()}
            log(f"ablation heart/{variant}: " + ", ".join(
                f"{n} AUC {v['roc_auc']['mean']:.4f}" for n, v in out[variant].items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(DISEASES), default=None)
    ap.add_argument("--fast", action="store_true", help="debug mode: tiny grids / fewer repeats")
    args = ap.parse_args()

    keys = [args.only] if args.only else list(DISEASES)
    results = {"meta": {"environment": environment_info(), "random_state": config.RANDOM_STATE,
                        "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
                        "cv_repeats": 1 if args.fast else config.CV_REPEATS, "fast_mode": args.fast,
                        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
               "diseases": {}}
    for k in keys:
        results["diseases"][k] = run_disease(k, fast=args.fast)
    results["meta"]["peak_rss_mb_main_process"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results["meta"]["total_training_wall_s"] = time.perf_counter() - T0

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / ("results_fast_debug.json" if args.fast else "results.json")
    if args.only and not args.fast and out.exists():          # merge single-disease re-runs
        prev = json.loads(out.read_text())
        prev["diseases"].update(results["diseases"])
        prev["meta"] = results["meta"]
        results = prev
    out.write_text(json.dumps(results, indent=2))
    log(f"results written to {out}")


if __name__ == "__main__":
    main()
