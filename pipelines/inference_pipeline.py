# ============================================================
# pipelines/inference_pipeline.py
# Pipeline Prefect : Collect new data → Predict → Monitor drift
#
# Usage :
#   python -m pipelines.inference_pipeline
# ============================================================

import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


@task(name="collect-recent-data", description="Collecte des dernières 48h de données")
def task_collect_recent(config_path: str, hours: int = 48) -> pd.DataFrame:
    logger = get_run_logger()
    from src.data.collect import load_config, fetch_openmeteo
    import httpx

    cfg = load_config(config_path)
    end   = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%d")

    om = cfg["openmeteo"]
    params = {
        "latitude":  om["latitude"],
        "longitude": om["longitude"],
        "start_date": start,
        "end_date":   end,
        "hourly":     om["variables"],
        "timezone":   "Europe/Paris",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get("https://api.open-meteo.com/v1/forecast", params=params)
        resp.raise_for_status()
        data = resp.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename_axis("datetime")
    df.columns = [f"meteo_{c}" for c in df.columns]

    logger.info(f"Données récentes : {len(df)} lignes")
    return df


@task(name="build-inference-features", description="Features engineering sur nouvelles données")
def task_inference_features(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    from src.features.build_features import (
        add_time_features, add_lag_features,
        add_rolling_features, add_differencing
    )
    TARGET = "meteo_temperature_2m"

    df = add_time_features(df)
    df = add_lag_features(df, TARGET, lags=[1, 2, 3, 6, 12, 24, 48])
    df = add_rolling_features(df, TARGET, windows=[6, 24])
    df = add_differencing(df, TARGET)
    df = df.dropna()

    logger.info(f"Features inférence : {df.shape}")
    return df


@task(name="load-model", description="Chargement du modèle de production MLflow")
def task_load_model(config_path: str):
    logger = get_run_logger()
    import joblib, mlflow.sklearn

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Essai MLflow registry d'abord
    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        model = mlflow.sklearn.load_model(
            f"models:/{cfg['mlflow']['model_name']}/Production"
        )
        logger.info("Modèle chargé depuis MLflow Registry (Production)")
    except Exception:
        # Fallback : fichier local
        local_path = Path(cfg["paths"]["models"]) / "best_model" / "model.pkl"
        if local_path.exists():
            model = joblib.load(local_path)
            logger.info(f"Modèle chargé localement : {local_path}")
        else:
            raise FileNotFoundError(
                "Aucun modèle disponible (ni MLflow ni local). "
                "Lancer d'abord : make train"
            )
    return model


@task(name="predict", description="Inférence : génération des prédictions")
def task_predict(model, df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    TARGET  = cfg["project"]["target"]
    HORIZON = cfg["project"]["horizon_hours"]
    feat_cols = [c for c in df.columns
                 if c != TARGET and not c.startswith("target")]

    preds = model.predict(df[feat_cols])
    result = pd.DataFrame({
        "datetime":           df.index,
        "predicted_temp_24h": preds,
        "lower":              preds - 2 * 1.5,   # ±2σ empirique
        "upper":              preds + 2 * 1.5,
    }).set_index("datetime")

    out = Path("reports") / f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H')}.parquet"
    result.to_parquet(out)
    logger.info(f"{len(result)} prédictions sauvegardées : {out}")
    return result


@task(name="check-drift", description="Détection drift Evidently vs données de référence")
def task_check_drift(current_df: pd.DataFrame, config_path: str) -> dict:
    logger = get_run_logger()
    from src.monitoring.drift import compute_drift_report
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Données de référence = dernières semaines du train
    ref_path = Path(cfg["paths"]["processed"]) / "features_train.parquet"
    if not ref_path.exists():
        logger.warning("Données de référence absentes, drift check ignoré")
        return {"skipped": True}

    reference = pd.read_parquet(ref_path).tail(7 * 24)   # 1 semaine de référence
    # Aligner les colonnes
    common_cols = [c for c in current_df.columns if c in reference.columns]
    if not common_cols:
        return {"skipped": True, "reason": "no common columns"}

    drift_result = compute_drift_report(
        reference=reference[common_cols],
        current=current_df[common_cols],
        output_dir=cfg["paths"]["monitoring"],
    )
    logger.info(f"  Dataset drift : {drift_result['dataset_drift']}")
    logger.info(f"  Features en drift : {drift_result['n_drifted_features']}")
    return drift_result


# ─── Flow principal ───────────────────────────────────────────

@flow(
    name="inference-pipeline",
    description="Inférence temps réel + monitoring drift",
    version="1.0.0",
)
def inference_pipeline(
    config_path: str = "configs/config.yaml",
    hours:       int = 48,
):
    logger = get_run_logger()
    logger.info("Démarrage du pipeline d'inférence")
    start = datetime.now()

    # 1. Collecter données récentes
    recent_raw = task_collect_recent(config_path, hours=hours)

    # 2. Features
    recent_feat = task_inference_features(recent_raw, config_path)

    # 3. Charger modèle
    model = task_load_model(config_path)

    # 4. Prédire
    predictions = task_predict(model, recent_feat, config_path)

    # 5. Check drift
    drift = task_check_drift(recent_feat, config_path)

    elapsed = (datetime.now() - start).seconds
    last_pred = float(predictions["predicted_temp_24h"].iloc[-1]) if len(predictions) > 0 else None

    create_markdown_artifact(
        key="inference-summary",
        markdown=f"""
# Inference Pipeline — Résumé

| Champ                  | Valeur |
|------------------------|--------|
| Prédictions générées   | `{len(predictions)}` |
| Dernière pred (T+24h)  | `{last_pred:.2f}°C` |
| Dataset drift          | `{drift.get('dataset_drift', 'N/A')}` |
| Features en drift      | `{drift.get('n_drifted_features', 'N/A')}` |
| Action requise         | `{drift.get('action_required', 'N/A')}` |
| Durée                  | `{elapsed}s` |

>  Re-entraînement recommandé si `action_required = RETRAIN`
        """,
        description="Résumé du pipeline d'inférence",
    )
    return {"predictions": predictions, "drift": drift}


if __name__ == "__main__":
    inference_pipeline()
