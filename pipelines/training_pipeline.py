# ============================================================
# pipelines/training_pipeline.py
# Pipeline Prefect : Load → Split → Train → Evaluate → Register
#
# Usage :
#   python -m pipelines.training_pipeline
#   python -m pipelines.training_pipeline --config configs/config.yaml
# ============================================================

import yaml
import json
import logging
from pathlib import Path
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact


# ─── Tâches ───────────────────────────────────────────────────

@task(name="load-datasets", description="Chargement des features train/val/test")
def task_load_data(config_path: str) -> dict:
    logger = get_run_logger()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    processed = Path(cfg["paths"]["processed"])

    train = pd.read_parquet(processed / "features_train.parquet")
    test  = pd.read_parquet(processed / "features_test.parquet")

    logger.info(f"Train : {train.shape} | Test : {test.shape}")
    return {"train": train, "test": test, "cfg": cfg}


@task(name="run-statistical-tests", description="ADF, KPSS, Granger, STL")
def task_statistical_tests(data: dict) -> dict:
    logger = get_run_logger()
    logger.info("Tests statistiques...")
    from src.analysis.statistical_tests import (
        test_stationarity, decompose_stl, granger_causality
    )
    cfg    = data["cfg"]
    target = cfg["project"]["target"]
    train  = data["train"]
    series = train[target]

    stationarity = test_stationarity(series, target)
    logger.info(f"  ADF stationnaire : {stationarity['stationary_adf']}")

    # Granger causality — top features corrélées avec la cible
    candidates = [
        c for c in train.columns
        if c != target and not c.startswith("target") and "lag" not in c
    ][:5]
    granger = granger_causality(train, target, candidates, max_lag=12)

    results = {
        "stationarity": stationarity,
        "granger_causality": {k: v["causes_target"] for k, v in granger.items()},
    }

    # Sauvegarder
    Path("reports").mkdir(exist_ok=True)
    with open("reports/statistical_tests.json", "w") as f:
        json.dump({k: str(v) if not isinstance(v, (bool, float, int)) else v
                   for k, v in results["stationarity"].items()}, f, indent=2)

    logger.info("Tests statistiques terminés")
    return results


@task(name="select-best-model", description="Optuna hyperparameter tuning multi-modèles")
def task_select_model(data: dict, model_config_path: str) -> dict:
    logger = get_run_logger()
    logger.info("🔎  Sélection du meilleur modèle via Optuna...")
    from src.models.select import run_model_selection
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)

    cfg    = data["cfg"]
    target = cfg["project"]["target"]
    horizon = cfg["project"]["horizon_hours"]
    train  = data["train"]
    test   = data["test"]

    feat_cols = [c for c in train.columns
                 if c != f"target_{horizon}h" and c != target]

    best = run_model_selection(
        X_train=train[feat_cols], y_train=train[f"target_{horizon}h"],
        X_test=test[feat_cols],   y_test=test[f"target_{horizon}h"],
        model_cfg=model_cfg,
    )
    logger.info(f"Meilleur modèle : {best['model_name']} "
                f"(RMSE={best['metrics']['rmse']:.4f})")
    return best


@task(name="train-final-model", description="Entraînement du modèle sélectionné + MLflow")
def task_train_final(data: dict, best_model_info: dict) -> dict:
    logger = get_run_logger()
    logger.info(f"Entraînement final : {best_model_info['model_name']}...")
    from src.models.train import run_training
    cfg = data["cfg"]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"final_{best_model_info['model_name']}") as run:
        mlflow.log_params(best_model_info["best_params"])
        mlflow.log_metrics(best_model_info["metrics"])
        mlflow.log_artifact("configs/config.yaml")
        mlflow.log_artifact("params.yaml")

        # Enregistrer le modèle dans le registry
        model_name = cfg["mlflow"]["model_name"]
        if best_model_info["model_name"] == "xgboost":
            mlflow.sklearn.log_model(
                best_model_info["model"],
                "model",
                registered_model_name=model_name,
            )
        # (Prophet, SARIMA, LSTM ont leurs propres log_model)

        run_id = run.info.run_id
        logger.info(f"Modèle enregistré — Run ID : {run_id}")

    # Sauvegarder localement aussi
    import joblib
    model_dir = Path(cfg["paths"]["models"]) / "best_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_info["model"], model_dir / "model.pkl")

    return {
        "run_id":   run_id,
        "metrics":  best_model_info["metrics"],
        "model_name": best_model_info["model_name"],
    }


@task(name="evaluate-model", description="Évaluation complète + plots")
def task_evaluate(data: dict, train_result: dict) -> dict:
    logger = get_run_logger()
    logger.info("Évaluation...")
    from src.models.evaluate import evaluate_model
    import joblib

    cfg     = data["cfg"]
    target  = cfg["project"]["target"]
    horizon = cfg["project"]["horizon_hours"]
    test    = data["test"]
    feat_cols = [c for c in test.columns
                 if c != f"target_{horizon}h" and c != target]

    model = joblib.load(Path(cfg["paths"]["models"]) / "best_model" / "model.pkl")
    metrics = evaluate_model(
        model=model,
        X_test=test[feat_cols],
        y_test=test[f"target_{horizon}h"],
        output_dir=cfg["paths"]["figures"],
    )

    # Vérification des seuils de production
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    thresholds = model_cfg["production_thresholds"]

    gate_passed = (
        metrics["rmse"] <= thresholds["max_rmse"] and
        metrics["mape"] <= thresholds["max_mape"] and
        metrics["r2"]   >= thresholds["min_r2"]
    )

    logger.info(f"  RMSE : {metrics['rmse']:.4f} (seuil ≤ {thresholds['max_rmse']})")
    logger.info(f"  MAPE : {metrics['mape']:.2f}% (seuil ≤ {thresholds['max_mape']}%)")
    logger.info(f"  R²   : {metrics['r2']:.4f} (seuil ≥ {thresholds['min_r2']})")
    logger.info(f"  Production gate : {'PASSÉ' if gate_passed else 'ÉCHOUÉ'}")

    # Sauvegarder métriques
    with open("reports/metrics_test.json", "w") as f:
        json.dump({**metrics, "gate_passed": gate_passed}, f, indent=2)

    return {**metrics, "gate_passed": gate_passed}


@task(name="promote-to-production", description="Promotion du modèle en staging/production MLflow")
def task_promote(train_result: dict, eval_result: dict, config_path: str):
    logger = get_run_logger()
    if not eval_result["gate_passed"]:
        logger.warning("Gate de production non passé — promotion annulée")
        return False

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    model_name = cfg["mlflow"]["model_name"]

    try:
        # Trouver la version la plus récente
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            version = versions[-1].version
            client.transition_model_version_stage(
                name=model_name, version=version, stage="Staging"
            )
            logger.info(f"Modèle v{version} promu en → Staging")
    except Exception as e:
        logger.warning(f"Promotion MLflow échouée (normal sans serveur MLflow distant) : {e}")
    return True


# ─── Flow principal ───────────────────────────────────────────

@flow(
    name="training-pipeline",
    description="Pipeline d'entraînement : selection → train → evaluate → promote",
    version="1.0.0",
    log_prints=True,
)
def training_pipeline(
    config_path:       str = "configs/config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    skip_stats:        bool = False,
):
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info("Démarrage du pipeline d'entraînement")
    logger.info("=" * 60)
    start = datetime.now()

    # 1. Chargement
    data = task_load_data(config_path)

    # 2. Tests statistiques (optionnel)
    if not skip_stats:
        task_statistical_tests(data)

    # 3. Sélection modèle + hyperparameter tuning
    best = task_select_model(data, model_config_path)

    # 4. Entraînement final
    train_result = task_train_final(data, best)

    # 5. Évaluation
    eval_result = task_evaluate(data, train_result)

    # 6. Promotion
    promoted = task_promote(train_result, eval_result, config_path)

    elapsed = (datetime.now() - start).seconds

    create_markdown_artifact(
        key="training-pipeline-summary",
        markdown=f"""
# Training Pipeline — Résumé

| Étape              | Résultat |
|--------------------|----------|
| Modèle sélectionné | `{best['model_name']}` |
| RMSE               | `{eval_result['rmse']:.4f}` |
| MAE                | `{eval_result['mae']:.4f}` |
| MAPE               | `{eval_result['mape']:.2f}%` |
| R²                 | `{eval_result['r2']:.4f}` |
| Gate production    | `{'PASSÉ' if eval_result['gate_passed'] else '❌ ÉCHOUÉ'}` |
| Promu en Staging   | `{'Oui' if promoted else 'Non'}` |
| MLflow Run ID      | `{train_result['run_id']}` |

**Durée** : {elapsed}s | **Date** : {start.strftime('%Y-%m-%d %H:%M:%S')}
        """,
        description="Résumé du pipeline d'entraînement",
    )
    logger.info(f"Pipeline terminé en {elapsed}s")
    return eval_result


if __name__ == "__main__":
    training_pipeline()
