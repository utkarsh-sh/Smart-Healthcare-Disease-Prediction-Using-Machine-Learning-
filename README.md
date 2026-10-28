# BT60: Smart Healthcare Disease Prediction Using Machine Learning

Capstone Phase III prototype, IILM University.
Team: Utkarsh Kishor (CS-2341383), Ravi Raj (CS-2341382), Nand Kishor Kumar (CS-23411301), Krishna Kumar (CS-2341401).

Three machine-learning risk models (diabetes, heart disease, breast cancer) trained with one shared,
leak-free protocol and served through a validated REST API, a browser UI and an SQLite audit log.

> Educational prototype. Outputs are statistical risk estimates from public research datasets
> and are **not** a medical diagnosis.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=.                 # (Windows PowerShell: $env:PYTHONPATH = ".")

python -m src.train                 # ~4 min: trains all modules, writes models/ and reports/results.json
python -m app.api                   # open http://127.0.0.1:5000
python -m pytest --cov=src --cov=app   # full test suite
python scripts/benchmark.py         # HTTP latency / throughput  -> reports/benchmark.json
python scripts/make_figures.py      # figures                    -> reports/figures/fig_*.png
python scripts/screenshots.py       # UI screenshots (needs `playwright install chromium`)
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/registry.py` | Single source of truth: feature specs, valid ranges, sentinel codes, loaders |
| `src/transformers.py` | `SentinelToNaN`, kept inside the saved pipeline (no train/serve skew) |
| `src/modeling.py`, `src/metrics.py` | Pipelines, model zoo, grids, ensemble; metrics, bootstrap CIs, screening threshold |
| `src/train.py` | Complete experimental protocol and artefact export |
| `src/predictor.py` | Validation, prediction, explanation, batch scoring |
| `src/store.py` | SQLite audit log |
| `app/` | Flask API (`api.py`), HTML/CSS/JS front end |
| `tests/` | pytest suite (data, inference, API, storage, metrics, training smoke test) |
| `scripts/` | benchmark, figures, screenshots, result tables, scenario check, run comparison, ensemble profile |
| `data/raw/` | Source CSVs (Pima diabetes, UCI Cleveland heart); breast cancer ships with scikit-learn |
| `models/`, `reports/` | Trained artefacts, results, figures |
| `docs/DATA_CODEBOOK.md` | Dataset notes and the defects found in them |

## Experimental protocol (same for every disease)

1. Stratified 80/20 split, seed 42; the 20% hold-out is used only after the model is chosen (no decision depends on it).
2. Baseline: majority class + 5 models, 5-fold CV x 3 repeats.
3. Tuning: `GridSearchCV` (5-fold, ROC-AUC).
4. Selection on different CV folds (seed 7), including a soft-voting ensemble; ties within 0.002 AUC go to the simpler model.
5. Hold-out evaluation with bootstrap 95% CIs, a screening threshold (>= 90% out-of-fold recall), permutation importance.
6. Stability over 10 further random splits, plus preprocessing ablations.

## API

| Route | Description |
| --- | --- |
| `GET /api/health` | Liveness, loaded modules, database state |
| `GET /api/diseases`, `GET /api/diseases/<key>` | Feature specs, thresholds, metrics, typical examples |
| `POST /api/predict/<key>` | One prediction + top contributing factors (`?explain=false` to skip) |
| `POST /api/predict/<key>/batch` | Up to 500 rows, per-row errors |
| `GET /api/history` | Recent predictions from SQLite |

```bash
curl -s -X POST http://127.0.0.1:5000/api/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":4,"Glucose":148,"BloodPressure":60,"SkinThickness":27,"Insulin":318,"BMI":30.9,"DiabetesPedigreeFunction":0.15,"Age":29}'
```

## Adding another disease

Add a `DiseaseSpec` to `src/registry.py` (features, ranges, loader), run `python -m src.train --only <key>`.
The API, validation and UI pick it up automatically.
