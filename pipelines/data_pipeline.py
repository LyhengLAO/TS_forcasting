# ============================================================
# pipelines/data_pipeline.py
# Pipeline Prefect : Collecte → Validation → Preprocessing → Features
#
# Usage :
#   python -m pipelines.data_pipeline          # run local
#   prefect deployment run data-pipeline/local  # via Prefect UI
# ============================================================

import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.artifacts import create_markdown_artifact

# ─── Tâches atomiques ─────────────────────────────────────────

@task(
    name="collect-openmeteo",
    retries=3,
    retry_delay_seconds=30,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
    description="Collecte données météo OpenMeteo (sans clé API)",
)
def task_collect_weather(config_path: str) -> str:
    logger = get_run_logger()
    logger.info("Collecte OpenMeteo...")
    from src.data.collect import load_config, fetch_openmeteo
    cfg = load_config(config_path)
    df = fetch_openmeteo(cfg)
    out = Path(cfg["paths"]["raw"]) / "weather_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    logger.info(f"Météo sauvegardée : {out} ({len(df)} lignes)")
    return str(out)


@task(
    name="collect-yfinance",
    retries=3,
    retry_delay_seconds=20,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
    description="Collecte données financières via yfinance (sans clé API)",
)
def task_collect_finance(config_path: str) -> str:
    logger = get_run_logger()
    logger.info("Collecte yfinance...")
    from src.data.collect import load_config, fetch_yfinance
    cfg = load_config(config_path)
    df = fetch_yfinance(cfg)
    out = Path(cfg["paths"]["raw"]) / "finance_raw.parquet"
    df.to_parquet(out)
    logger.info(f"Finance sauvegardée : {out} ({len(df)} lignes)")
    return str(out)


@task(
    name="validate-data",
    retries=1,
    description="Validation Great Expectations des données brutes",
)
def task_validate(weather_path: str, finance_path: str, config_path: str) -> dict:
    logger = get_run_logger()
    logger.info("🔍  Validation des données...")
    import pandas as pd
    from src.data.validate import validate_weather
    df_weather = pd.read_parquet(weather_path)
    results = validate_weather(df_weather)
    failed = [k for k, v in results.items() if not v]
    if failed:
        raise ValueError(f"Validation échouée sur : {failed}")
    logger.info(f"Validation OK ({len(results)} checks passés)")

    # Sauvegarder le rapport de validation
    import json
    with open("reports/validation_report.json", "w") as f:
        json.dump({k: bool(v) for k, v in results.items()}, f, indent=2)
    return results


@task(
    name="preprocess-data",
    description="Nettoyage, outliers, imputation temporelle",
)
def task_preprocess(config_path: str) -> str:
    logger = get_run_logger()
    logger.info("🧹  Preprocessing...")
    from src.data.preprocess import preprocess
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    df = preprocess(cfg)
    logger.info(f"Preprocessing OK : {df.shape}")
    return str(Path(cfg["paths"]["interim"]) / "merged_clean.parquet")


@task(
    name="build-features",
    description="Feature engineering : lags, rolling stats, encodages cycliques",
)
def task_features(config_path: str) -> tuple:
    logger = get_run_logger()
    logger.info("Feature engineering...")
    from src.features.build_features import build_features
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train, test = build_features(cfg)
    logger.info(f"Features OK — Train: {train.shape} | Test: {test.shape}")
    return str(Path(cfg["paths"]["processed"]) / "features_train.parquet"), \
           str(Path(cfg["paths"]["processed"]) / "features_test.parquet")


# ─── Flow principal ───────────────────────────────────────────

@flow(
    name="data-pipeline",
    description="Pipeline complet de données : collecte → features",
    version="1.0.0",
    log_prints=True,
)
def data_pipeline(config_path: str = "configs/config.yaml"):
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("Démarrage du pipeline de données")
    logger.info("=" * 60)

    start_time = datetime.now()

    # Étape 1 : Collecte (parallèle)
    weather_future = task_collect_weather.submit(config_path)
    finance_future = task_collect_finance.submit(config_path)
    weather_path   = weather_future.result()
    finance_path   = finance_future.result()

    # Étape 2 : Validation
    task_validate(weather_path, finance_path, config_path)

    # Étape 3 : Preprocessing
    task_preprocess(config_path)

    # Étape 4 : Feature engineering
    train_path, test_path = task_features(config_path)

    elapsed = (datetime.now() - start_time).seconds

    # Artifact Prefect (visible dans l'UI)
    create_markdown_artifact(
        key="data-pipeline-summary",
        markdown=f"""
# Data Pipeline — Résumé

| Étape        | Statut | Sortie |
|--------------|--------|--------|
| Collecte météo  | ✅ | `{weather_path}` |
| Collecte finance | ✅ | `{finance_path}` |
| Validation       | ✅ | `reports/validation_report.json` |
| Preprocessing    | ✅ | `data/interim/merged_clean.parquet` |
| Features         | ✅ | `{train_path}` |

**Durée totale** : {elapsed}s  
**Exécuté le** : {start_time.strftime('%Y-%m-%d %H:%M:%S')}
        """,
        description="Résumé du pipeline de données",
    )
    logger.info(f"Pipeline terminé en {elapsed}s")
    return {"train_path": train_path, "test_path": test_path}


if __name__ == "__main__":
    data_pipeline()
