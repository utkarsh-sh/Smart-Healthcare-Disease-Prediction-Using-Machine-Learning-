"""Expected-versus-observed scenario check against the real API (Flask test client, throw-away database).

    PYTHONPATH=. python scripts/scenario_check.py   ->  reports/scenarios.json and reports/scenarios.md

Every row's "observed" column is read from the live response; "pass" is computed, not typed.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.api import create_app
from src import config

tmp = Path(tempfile.mkdtemp()) / "scenarios.db"
app = create_app({"DB_PATH": str(tmp), "TESTING": True})
client = app.test_client()
P = app.extensions["predictor"]
EX = {k: P.meta[k]["examples"] for k in P.available()}
rows: list[dict] = []


def record(sid, scenario, expected, observed, ok):
    rows.append({"id": sid, "scenario": scenario, "expected": expected, "observed": observed, "pass": bool(ok)})


def history_total():
    return sum(client.get("/api/history?limit=1").get_json()["counts"].values())


def post(path, payload=None, raw=None):
    if raw is not None:
        return client.post(path, data=raw, content_type="application/json")
    return client.post(path, json=payload)


# S1 health ------------------------------------------------------------------------------------
r = client.get("/api/health"); j = r.get_json()
record("S1", "GET /api/health", "200, status ok, 3 modules, database ok",
       f"{r.status_code}, status {j['status']}, modules {j['modules']}, database {j['database']}",
       r.status_code == 200 and j["status"] == "ok" and len(j["modules"]) == 3 and j["database"] == "ok")

# S2 catalogue order ---------------------------------------------------------------------------
r = client.get("/api/diseases"); names = [d["key"] for d in r.get_json()]
record("S2", "GET /api/diseases", "200, modules in registry order diabetes, heart, breast",
       f"{r.status_code}, {', '.join(names)}", r.status_code == 200 and names == ["diabetes", "heart", "breast"])

# S3 / S4 typical cases ------------------------------------------------------------------------
r = post("/api/predict/heart", EX["heart"]["positive"]["values"]); j = r.get_json()
record("S3", "Heart module, typical positive case", "200, risk above the 0.5 line, screening flag on",
       f"{r.status_code}, risk {j['probability']:.3f}, flag {j['screening_flag']}, {len(j['top_factors'])} factors",
       r.status_code == 200 and j["probability"] > 0.5 and j["screening_flag"] is True and len(j["top_factors"]) == 3)
r = post("/api/predict/heart", EX["heart"]["negative"]["values"]); j = r.get_json()
record("S4", "Heart module, typical negative case", "200, risk below the screening line, flag off",
       f"{r.status_code}, risk {j['probability']:.3f}, flag {j['screening_flag']}",
       r.status_code == 200 and j["probability"] < j["screening_threshold"] and j["screening_flag"] is False)

base = dict(EX["diabetes"]["positive"]["values"])

# S5-S8 validation -----------------------------------------------------------------------------
n0 = history_total()
bad = dict(base, Glucose=9999)
r = post("/api/predict/diabetes", bad); j = r.get_json()
record("S5", "Diabetes, glucose = 9999", "422 with a field message naming the range",
       f"{r.status_code}, Glucose: {j['fields'].get('Glucose')}", r.status_code == 422 and "Must be between" in j["fields"].get("Glucose", ""))
miss = {k: v for k, v in base.items() if k != "Age"}
r = post("/api/predict/diabetes", miss); j = r.get_json()
record("S6", "Diabetes, required Age omitted", "422, Age: Required",
       f"{r.status_code}, Age: {j['fields'].get('Age')}", r.status_code == 422 and j["fields"].get("Age") == "Required.")
r = post("/api/predict/diabetes", dict(base, Age=40.5)); j = r.get_json()
record("S7", "Diabetes, Age = 40.5", "422, whole number required",
       f"{r.status_code}, Age: {j['fields'].get('Age')}", r.status_code == 422 and "whole number" in j["fields"].get("Age", ""))
raw = json.dumps(dict(base, BMI=0.0)).replace('"BMI": 0.0', '"BMI": NaN')
r = post("/api/predict/diabetes", raw=raw); j = r.get_json()
record("S8", "Diabetes, BMI = NaN in the JSON body", "422, value must be finite",
       f"{r.status_code}, BMI: {j['fields'].get('BMI')}", r.status_code == 422 and "finite" in j["fields"].get("BMI", ""))
r = post("/api/predict/diabetes", dict(base, Cholesterol=200)); j = r.get_json()
record("S9", "Diabetes, unknown field Cholesterol", "422, unknown field rejected",
       f"{r.status_code}, fields {sorted(j['fields'])}", r.status_code == 422 and "Cholesterol" in j["fields"])
n1 = history_total()
record("S10", "Rejected requests S5 to S9 are not logged", "history count unchanged",
       f"{n0} before, {n1} after", n0 == n1)

# S11-S12 warnings -----------------------------------------------------------------------------
r = post("/api/predict/diabetes", {k: v for k, v in base.items() if k != "Insulin"}); j = r.get_json()
record("S11", "Diabetes, optional insulin left blank", "200 plus a 'not provided' warning",
       f"{r.status_code}, risk {j['probability']:.3f}, warnings {len(j['warnings'])}: {j['warnings'][0] if j['warnings'] else ''}",
       r.status_code == 200 and any("not provided" in w for w in j["warnings"]))
r = post("/api/predict/diabetes", dict(base, Insulin=880)); j = r.get_json()
record("S12", "Diabetes, valid but unusual insulin = 880", "200 plus an out-of-range-for-training warning",
       f"{r.status_code}, {j['warnings'][0] if j['warnings'] else 'no warning'}",
       r.status_code == 200 and any("outside the range typical" in w for w in j["warnings"]))

# S13-S16 protocol-level errors ----------------------------------------------------------------
r = post("/api/predict/asthma", base)
record("S13", "Unknown module asthma", "404 JSON error", f"{r.status_code}, {r.get_json().get('error')}", r.status_code == 404)
r = post("/api/predict/diabetes", raw="{not json")
record("S14", "Malformed JSON body", "400 JSON error", f"{r.status_code}, {r.get_json().get('fields')}", r.status_code == 400)
r = post("/api/predict/diabetes", raw="x" * 1_200_000)
record("S15", "Body larger than 1 MB", "413 JSON error", f"{r.status_code}", r.status_code == 413)
r = client.get("/api/predict/heart")
record("S16", "GET on a POST-only route", "405 JSON error", f"{r.status_code}, {r.get_json().get('error')}", r.status_code == 405)

# S17-S19 batch --------------------------------------------------------------------------------
r = post("/api/predict/diabetes/batch", {"rows": [base] * 501})
record("S17", "Batch of 501 rows", "400, limit of 500 enforced", f"{r.status_code}, {r.get_json().get('fields')}", r.status_code == 400)
r = post("/api/predict/diabetes/batch", {"rows": [base] * 500}); j = r.get_json()
record("S18", "Batch of 500 rows", "200, one result per row",
       f"{r.status_code}, {len(j.get('results', []))} results", r.status_code == 200 and len(j.get("results", [])) == 500)

# S19 logging ----------------------------------------------------------------------------------
before = history_total()
for _ in range(3):
    post("/api/predict/breast", EX["breast"]["positive"]["values"])
after = history_total()
record("S19", "Three valid breast-cancer predictions", "history grows by exactly 3", f"{before} before, {after} after", after - before == 3)


# S20-S21 consistency between what the UI/API serves and what the experiment recorded ------------
R = json.loads((config.REPORTS_DIR / "results.json").read_text())["diseases"]
served = client.get("/api/diseases").get_json()
mismatch, compared = [], 0
for d in served:
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        compared += 1
        if abs(d["metrics"]["test"][m] - R[d["key"]]["test"][m]) > 1e-12:
            mismatch.append(f"{d['key']}.{m}")
record("S20", "Hold-out metrics shown by the API (and UI) versus results.json", "identical for every module and metric",
       f"{compared - len(mismatch)} of {compared} values identical" + (f"; differ: {mismatch}" if mismatch else ""), not mismatch)
bad = [d["key"] for d in served
       if abs(d["thresholds"]["screening"] - R[d["key"]]["screening"]["threshold"]) > 1e-12
       or d["selected_model"] != R[d["key"]]["selected_model"]]
record("S21", "Served screening line and model name versus results.json", "identical for every module",
       f"{len(served) - len(bad)} of {len(served)} modules identical", not bad)

summary = {"scenarios": len(rows), "passed": sum(r["pass"] for r in rows), "model_version": config.MODEL_VERSION}
(config.REPORTS_DIR / "scenarios.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
md = ["| ID | Scenario | Expected | Observed | Result |", "| --- | --- | --- | --- | --- |"]
for r in rows:
    md.append(f"| {r['id']} | {r['scenario']} | {r['expected']} | {r['observed']} | {'Pass' if r['pass'] else 'FAIL'} |")
md.append("")
md.append(f"{summary['passed']} of {summary['scenarios']} scenarios passed.")
(config.REPORTS_DIR / "scenarios.md").write_text("\n".join(md))
print("\n".join(md))
