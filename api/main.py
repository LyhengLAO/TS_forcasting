# ============================================================
# api/main.py
# API FastAPI de prédiction de séries temporelles
#
# Endpoints :
#   GET  /health         → état de l'API + modèle
#   POST /predict        → prédiction T+24h
#   POST /predict/batch  → prédictions batch (max 168)
#   GET  /metrics        → métriques Prometheus
#   GET  /model/info     → infos modèle chargé
#   GET  /docs           → documentation Swagger (auto)
#
# Usage :
#   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)

from api.schemas import (
    ForecastRequest, ForecastResponse,
    BatchForecastRequest, BatchForecastResponse,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ─── Métriques Prometheus ─────────────────────────────────────

REQUEST_COUNT    = Counter(
    "predict_requests_total",
    "Nombre total de requêtes /predict",
    ["endpoint", "status"],
)
PREDICT_LATENCY  = Histogram(
    "predict_latency_seconds",
    "Latence des prédictions (secondes)",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
MODEL_RMSE_LIVE  = Gauge(
    "model_rmse_live",
    "RMSE en production (fenêtre glissante)",
)
DRIFT_SHARE      = Gauge(
    "drift_share_features",
    "Part des features en drift (0-1)",
)
DRIFT_ALERTS     = Counter(
    "drift_alerts_total",
    "Nombre d'alertes drift déclenchées",
)
MODEL_LOADED     = Gauge(
    "model_loaded",
    "1 si le modèle est chargé, 0 sinon",
)

# ─── État global de l'application ─────────────────────────────

APP_STATE = {
    "model":         None,
    "model_name":    "unknown",
    "model_version": "unknown",
    "metadata":      {},
    "feature_names": [],
    "loaded_at":     None,
}


# ─── Chargement du modèle ─────────────────────────────────────

def _load_model_from_registry(cfg: dict):
    """Tente de charger depuis MLflow Registry."""
    import mlflow.sklearn
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    return mlflow.sklearn.load_model(
        f"models:/{cfg['mlflow']['model_name']}/Production"
    )


def _load_model_local(cfg: dict):
    """Charge le modèle depuis le fichier local."""
    import joblib
    model_path = Path(cfg["paths"]["models"]) / "best_model" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle local introuvable : {model_path}\n"
            "Lancer d'abord : make train"
        )
    return joblib.load(model_path)


def load_model_and_metadata(cfg: dict) -> None:
    """Charge le modèle et les métadonnées dans APP_STATE."""
    # 1. Essai MLflow registry
    try:
        model = _load_model_from_registry(cfg)
        APP_STATE["model_version"] = "mlflow_production"
        log.info("✅ Modèle chargé depuis MLflow Registry (Production)")
    except Exception as e:
        log.warning(f"MLflow Registry indisponible ({e}) → chargement local")
        model = _load_model_local(cfg)
        APP_STATE["model_version"] = "local"

    APP_STATE["model"]     = model
    APP_STATE["loaded_at"] = datetime.utcnow().isoformat()

    # Métadonnées
    meta_path = Path(cfg["paths"]["models"]) / "best_model" / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        APP_STATE["metadata"]      = meta
        APP_STATE["model_name"]    = meta.get("model_name", "unknown")
        APP_STATE["feature_names"] = meta.get("feature_names", [])
        log.info(f"  Modèle    : {APP_STATE['model_name']}")
        log.info(f"  Features  : {len(APP_STATE['feature_names'])}")
        log.info(f"  RMSE test : {meta.get('test_metrics', {}).get('rmse', 'N/A')}")

    # Feature names depuis fichier texte
    feat_path = Path(cfg["paths"]["processed"]) / "feature_names.txt"
    if feat_path.exists() and not APP_STATE["feature_names"]:
        with open(feat_path) as f:
            APP_STATE["feature_names"] = [l.strip() for l in f if l.strip()]

    MODEL_LOADED.set(1)


# ─── Lifespan (startup/shutdown) ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chargement du modèle au démarrage."""
    try:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        load_model_and_metadata(cfg)
        log.info("🚀 API démarrée avec succès")
    except Exception as e:
        log.error(f"❌ Erreur au démarrage : {e}")
        MODEL_LOADED.set(0)
    yield
    log.info("🛑 API arrêtée")


# ─── Application FastAPI ───────────────────────────────────────

app = FastAPI(
    title="ts-forecast API",
    description="API de prévision de température (T+24h) basée sur données météo/finance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS (adapter en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Middleware de logging ─────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed  = time.perf_counter() - start
    log.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed*1000:.1f}ms)"
    )
    return response


# ─── Endpoints ────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"],
)
def health():
    return HealthResponse(
        status="ok" if APP_STATE["model"] is not None else "degraded",
        model_loaded=APP_STATE["model"] is not None,
        timestamp=datetime.utcnow().isoformat(),
        version=app.version,
    )


@app.get(
    "/model/info",
    summary="Informations sur le modèle chargé",
    tags=["Modèle"],
)
def model_info():
    if APP_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    meta = APP_STATE["metadata"]
    return {
        "model_name":    APP_STATE["model_name"],
        "model_version": APP_STATE["model_version"],
        "loaded_at":     APP_STATE["loaded_at"],
        "n_features":    len(APP_STATE["feature_names"]),
        "test_rmse":     meta.get("test_metrics", {}).get("rmse"),
        "test_mape":     meta.get("test_metrics", {}).get("mape"),
        "test_r2":       meta.get("test_metrics", {}).get("r2"),
        "horizon_hours": meta.get("horizon_h", 24),
    }


@app.post(
    "/predict",
    response_model=ForecastResponse,
    summary="Prédiction température T+24h",
    tags=["Prédiction"],
    status_code=status.HTTP_200_OK,
)
def predict(req: ForecastRequest):
    REQUEST_COUNT.labels(endpoint="/predict", status="attempt").inc()

    if APP_STATE["model"] is None:
        REQUEST_COUNT.labels(endpoint="/predict", status="error_no_model").inc()
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    t0 = time.perf_counter()

    try:
        # Construire le vecteur de features
        features = req.features

        if APP_STATE["feature_names"]:
            # Ordre des features contraint par le modèle entraîné
            X = pd.DataFrame(
                [[features.get(f, 0.0) for f in APP_STATE["feature_names"]]],
                columns=APP_STATE["feature_names"],
            )
        else:
            X = pd.DataFrame([features])

        # Prédiction
        pred_raw = APP_STATE["model"].predict(X)
        pred     = float(pred_raw[0])

        # Intervalle de confiance empirique (±2σ calibré sur résidus validation)
        # En production : utiliser la quantile regression ou conformal prediction
        sigma   = APP_STATE["metadata"].get("test_metrics", {}).get("rmse", 1.5)
        lower   = pred - 2.0 * sigma
        upper   = pred + 2.0 * sigma

        latency_ms = (time.perf_counter() - t0) * 1000
        PREDICT_LATENCY.observe(latency_ms / 1000)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()

        return ForecastResponse(
            datetime_utc=req.datetime_utc,
            predicted_temp_24h=round(pred, 3),
            confidence_lower=round(lower, 3),
            confidence_upper=round(upper, 3),
            model_version=APP_STATE["model_version"],
            prediction_latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        log.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchForecastResponse,
    summary="Prédictions batch",
    tags=["Prédiction"],
)
def predict_batch(req: BatchForecastRequest):
    REQUEST_COUNT.labels(endpoint="/predict/batch", status="attempt").inc()

    if APP_STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    predictions = []
    for single_req in req.requests:
        try:
            resp = predict(single_req)
            predictions.append(resp)
        except HTTPException:
            predictions.append(None)   # échec partiel toléré

    valid_preds = [p for p in predictions if p is not None]

    return BatchForecastResponse(
        predictions=valid_preds,
        n_predictions=len(valid_preds),
        model_version=APP_STATE["model_version"],
    )


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Métriques Prometheus",
    tags=["Monitoring"],
    include_in_schema=False,
)
def metrics():
    """Endpoint Prometheus — scrappé toutes les 15s par Prometheus."""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/model/reload",
    summary="Recharger le modèle (sans redémarrer l'API)",
    tags=["Modèle"],
)
def reload_model():
    """Permet de mettre à jour le modèle en production sans downtime."""
    try:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        load_model_and_metadata(cfg)
        return {
            "status": "reloaded",
            "model_name": APP_STATE["model_name"],
            "loaded_at": APP_STATE["loaded_at"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rechargement échoué : {e}")


# ─── Gestionnaire d'erreurs global ────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Erreur non gérée : {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur", "type": type(exc).__name__},
    )
