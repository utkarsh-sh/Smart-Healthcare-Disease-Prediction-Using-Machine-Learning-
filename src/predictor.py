"""Inference service: loads trained artefacts, validates input, predicts and explains.

This is the single entry point used by the REST API (and usable directly from notebooks).
"""
from __future__ import annotations

import json
import math
import numbers
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .registry import DISEASES, get_spec


class ValidationError(Exception):
    """Raised when a payload does not satisfy the feature specification."""

    def __init__(self, errors: dict):
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


class DiseasePredictor:
    def __init__(self, models_dir: Path | str = config.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models: dict = {}
        self.meta: dict = {}
        for key in DISEASES:
            model_file = self.models_dir / f"{key}_model.joblib"
            meta_file = self.models_dir / f"{key}_meta.json"
            if model_file.exists() and meta_file.exists():
                self.models[key] = joblib.load(model_file)
                self.meta[key] = json.loads(meta_file.read_text())

    # ------------------------------------------------------------------ catalogue
    def available(self) -> list[str]:
        return [k for k in DISEASES if k in self.models]          # registry order

    def _require(self, key: str) -> None:
        get_spec(key)                        # KeyError for unknown disease modules
        if key not in self.models:
            raise KeyError(f"Model for '{key}' is not trained yet. Run `python -m src.train`.")

    def describe(self, key: str) -> dict:
        self._require(key)
        m = self.meta[key]
        return {
            "key": key, "name": m["name"], "description": m["description"],
            "positive_label": m["positive_label"], "negative_label": m["negative_label"],
            "features": m["features"], "selected_model": m["selected_model"], "version": m["version"],
            "trained_at": m["trained_at"], "thresholds": m["thresholds"], "metrics": m["metrics"],
            "n_train": m["n_train"], "n_test": m["n_test"], "source": m["source"],
            "population_note": m["population_note"], "examples": m["examples"],
            "train_stats": {k: {kk: vv for kk, vv in v.items() if kk in ("p1", "p99", "min", "max", "median")}
                            for k, v in m["train_stats"].items()},
        }

    # ------------------------------------------------------------------ validation
    def validate(self, key: str, payload) -> tuple[dict, list[str]]:
        """Return (clean_row, warnings) or raise :class:`ValidationError` listing every problem."""
        self._require(key)
        spec = get_spec(key)
        stats = self.meta[key]["train_stats"]
        if not isinstance(payload, dict):
            raise ValidationError({"_payload": "Request body must be a JSON object of feature values."})

        errors: dict = {}
        warnings: list[str] = []
        known = set(spec.feature_names)
        for extra in sorted(set(payload) - known):
            errors[extra] = "Unknown field."
        row: dict = {}
        for f in spec.features:
            raw = payload.get(f.name)
            if raw is None:
                if f.allow_missing:
                    row[f.name] = np.nan
                    warnings.append(f"{f.label} not provided: replaced by the training median/mode.")
                else:
                    errors[f.name] = "Required."
                continue
            if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
                errors[f.name] = "Must be a number."
                continue
            val = float(raw)
            if math.isnan(val) or math.isinf(val):
                errors[f.name] = "Must be a finite number."
                continue
            if f.kind in ("binary", "category"):
                allowed = [c for c, _ in f.choices]
                if val not in allowed:
                    errors[f.name] = f"Must be one of {allowed}."
                    continue
            else:
                if f.kind == "integer" and val != int(val):
                    errors[f.name] = "Must be a whole number."
                    continue
                if val < f.min or val > f.max:
                    unit = f" {f.unit}" if f.unit else ""
                    errors[f.name] = f"Must be between {f.min:g} and {f.max:g}{unit}."
                    continue
                s = stats[f.name]
                if val < s["p1"] or val > s["p99"]:
                    warnings.append(f"{f.label} = {val:g} lies outside the range typical of the training data "
                                    f"({s['p1']:.4g} to {s['p99']:.4g}); the estimate may be less reliable.")
            row[f.name] = val
        if errors:
            raise ValidationError(errors)
        return row, warnings

    # ------------------------------------------------------------------ prediction
    def _frame(self, key: str, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=get_spec(key).feature_names).astype(float)

    def predict(self, key: str, payload, explain: bool = True, top_k: int = 3) -> dict:
        row, warnings = self.validate(key, payload)
        t0 = time.perf_counter()
        frame = self._frame(key, [row])
        proba = float(self.models[key].predict_proba(frame)[0, 1])
        meta = self.meta[key]
        thr = meta["thresholds"]
        result = {
            "disease": key, "disease_name": meta["name"],
            "probability": proba,
            "prediction": int(proba >= thr["default"]),
            "label": meta["positive_label"] if proba >= thr["default"] else meta["negative_label"],
            "threshold": thr["default"],
            "screening_threshold": thr["screening"],
            "screening_flag": bool(proba >= thr["screening"]),
            "warnings": warnings,
            "model": {"name": meta["selected_model"], "version": meta["version"]},
            "disclaimer": config.DISCLAIMER,
        }
        if explain:
            result["top_factors"] = self._explain(key, row, frame, proba, top_k)
        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def predict_batch(self, key: str, payloads: list) -> dict:
        """Vectorised scoring. Invalid rows are reported by index instead of failing the whole batch."""
        self._require(key)
        meta = self.meta[key]
        thr = meta["thresholds"]
        t0 = time.perf_counter()
        good_idx, good_rows, errors = [], [], {}
        for i, p in enumerate(payloads):
            try:
                row, _ = self.validate(key, p)
                good_idx.append(i)
                good_rows.append(row)
            except ValidationError as exc:
                errors[i] = exc.errors
        results = []
        if good_rows:
            probas = self.models[key].predict_proba(self._frame(key, good_rows))[:, 1]
            for i, p in zip(good_idx, probas):
                results.append({"index": i, "probability": float(p), "prediction": int(p >= thr["default"]),
                                "screening_flag": bool(p >= thr["screening"])})
        return {"disease": key, "n_rows": len(payloads), "n_scored": len(results), "n_invalid": len(errors),
                "results": results, "errors": {str(k): v for k, v in errors.items()},
                "latency_ms": (time.perf_counter() - t0) * 1000, "model": {"name": meta["selected_model"],
                                                                           "version": meta["version"]}}

    # ------------------------------------------------------------------ explanation
    def _explain(self, key: str, row: dict, frame: pd.DataFrame, proba: float, top_k: int) -> list[dict]:
        """Model-agnostic occlusion attribution.

        Each provided feature is replaced, one at a time, by its training median (numeric) or mode
        (categorical). The change in predicted probability is the feature's local contribution.
        It is a *what-if* explanation, not a causal statement, and not SHAP.
        """
        spec = get_spec(key)
        stats = self.meta[key]["train_stats"]
        present = [f for f in spec.features if not math.isnan(row[f.name])]
        variants = pd.concat([frame] * len(present), ignore_index=True)
        for i, f in enumerate(present):
            variants.loc[i, f.name] = stats[f.name]["baseline"]
        p_alt = self.models[key].predict_proba(variants)[:, 1]
        out = []
        for f, pa in zip(present, p_alt):
            delta = proba - float(pa)
            value = row[f.name]
            display = f"{value:g}"
            if f.choices:
                display = dict(f.choices).get(int(value), display)
            out.append({"feature": f.name, "label": f.label, "value": value, "display_value": display,
                        "reference": stats[f.name]["baseline"], "effect_pp": delta * 100,
                        "direction": "raises risk" if delta > 0 else "lowers risk" if delta < 0 else "neutral"})
        out.sort(key=lambda d: -abs(d["effect_pp"]))
        return out[:top_k]
