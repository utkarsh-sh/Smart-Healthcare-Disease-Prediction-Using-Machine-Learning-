"""Flask REST API + web UI for the Smart Healthcare Disease Prediction prototype.

Endpoints
    GET  /                              web UI
    GET  /api/health                    liveness + loaded modules + DB status
    GET  /api/diseases                  catalogue of disease modules (features, metrics, thresholds)
    GET  /api/diseases/<key>            one module
    POST /api/predict/<key>             single prediction (+ top contributing factors)
    POST /api/predict/<key>/batch       vectorised scoring of up to 500 rows
    GET  /api/history                   recent predictions from the SQLite audit log
"""
from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src import config
from src.predictor import DiseasePredictor, ValidationError
from src.store import PredictionStore

MAX_BATCH = 500


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"),
                static_folder=str(Path(__file__).parent / "static"))
    app.config.update(DB_PATH=str(config.DB_PATH), MODELS_DIR=str(config.MODELS_DIR),
                      MAX_CONTENT_LENGTH=1_000_000, LOG_PREDICTIONS=True)
    if test_config:
        app.config.update(test_config)

    predictor = DiseasePredictor(app.config["MODELS_DIR"])
    store = PredictionStore(app.config["DB_PATH"])
    app.extensions["predictor"] = predictor
    app.extensions["store"] = store

    def err(status: int, message: str, **extra):
        return jsonify({"error": message, **extra}), status

    def json_body():
        body = request.get_json(silent=True)
        if body is None:
            raise ValidationError({"_payload": "Request body must be valid JSON (Content-Type: application/json)."})
        return body

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template("index.html", disclaimer=config.DISCLAIMER)

    # ---------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        try:
            store.counts()
            db = "ok"
        except Exception as exc:                          # pragma: no cover - defensive
            db = f"error: {exc}"
        return jsonify({"status": "ok" if predictor.available() and db == "ok" else "degraded",
                        "modules": predictor.available(), "database": db,
                        "model_version": config.MODEL_VERSION})

    @app.get("/api/diseases")
    def diseases():
        return jsonify([predictor.describe(k) for k in predictor.available()])

    @app.get("/api/diseases/<key>")
    def disease(key):
        try:
            return jsonify(predictor.describe(key))
        except KeyError as exc:
            return err(404, str(exc.args[0]))

    @app.post("/api/predict/<key>")
    def predict(key):
        explain = request.args.get("explain", "true").lower() != "false"
        try:
            payload = json_body()
            result = predictor.predict(key, payload, explain=explain)
        except ValidationError as exc:
            return err(422 if "_payload" not in exc.errors else 400, "Invalid input.", fields=exc.errors)
        except KeyError as exc:
            return err(404, str(exc.args[0]))
        if app.config["LOG_PREDICTIONS"]:
            result["history_id"] = store.add(result, payload)
        return jsonify(result)

    @app.post("/api/predict/<key>/batch")
    def predict_batch(key):
        try:
            body = json_body()
            rows = body.get("rows") if isinstance(body, dict) else None
            if not isinstance(rows, list) or not rows:
                raise ValidationError({"rows": "Provide a non-empty list under the key 'rows'."})
            if len(rows) > MAX_BATCH:
                raise ValidationError({"rows": f"At most {MAX_BATCH} rows per request."})
            return jsonify(predictor.predict_batch(key, rows))
        except ValidationError as exc:
            return err(400, "Invalid input.", fields=exc.errors)
        except KeyError as exc:
            return err(404, str(exc.args[0]))

    @app.get("/api/history")
    def history():
        try:
            limit = max(1, min(int(request.args.get("limit", 20)), 200))
        except ValueError:
            return err(400, "limit must be an integer.")
        return jsonify({"items": store.recent(limit, request.args.get("disease")),
                        "counts": store.counts()})

    # ---------------------------------------------------------------- errors (always JSON under /api)
    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            return err(exc.code or 500, exc.description or exc.name)
        return exc

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):                         # pragma: no cover - last-resort guard
        app.logger.exception("Unhandled error")
        return err(500, "Internal server error.")

    return app


if __name__ == "__main__":                                 # python -m app.api
    create_app().run(host="127.0.0.1", port=5000, debug=False)
