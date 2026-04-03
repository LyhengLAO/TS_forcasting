# ============================================================
# src/models/train.py
# Pipeline d'entraînement complet :
#   - Optuna hyperparameter tuning
#   - MLflow experiment tracking
#   - Multi-modèles : XGBoost, SARIMA, Prophet, LSTM
#   - Sauvegarde locale + MLflow Model Registry
#
# Usage :
#   python -m src.models.train
#   python -m src.models.train --model xgboost --trials 30
# ============================================================

import argparse
import json
import logging
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


# ─── Chargement des données ───────────────────────────────────

def load_datasets(cfg: dict) -> dict:
    """Charge train, val et test depuis data/processed/."""
    processed = Path(cfg["paths"]["processed"])
    required  = ["features_train.parquet", "features_val.parquet", "features_test.parquet"]
    for f in required:
        if not (processed / f).exists():
            raise FileNotFoundError(
                f"Fichier manquant : {processed / f}\n"
                "Lancer d'abord : les parties data"
            )

    train = pd.read_parquet(processed / "features_train.parquet")
    val   = pd.read_parquet(processed / "features_val.parquet")
    test  = pd.read_parquet(processed / "features_test.parquet")

    target  = cfg["project"]["target"]
    horizon = cfg["project"]["horizon_hours"]
    target_col  = f"target_{horizon}h"
    feat_cols   = [c for c in train.columns
                   if c != target_col and c != target]

    log.info(f"Train : {train.shape} | Val : {val.shape} | Test : {test.shape}")
    log.info(f"Features : {len(feat_cols)} | Target : {target_col}")

    return {
        "X_train": train[feat_cols], "y_train": train[target_col],
        "X_val":   val[feat_cols],   "y_val":   val[target_col],
        "X_test":  test[feat_cols],  "y_test":  test[target_col],
        "feat_cols": feat_cols,
        "target_col": target_col,
    }


# ─── Métriques ────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - y_true.mean()) ** 2)
    r2   = float(1 - ss_r / (ss_t + 1e-8))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# ─── XGBoost ──────────────────────────────────────────────────

def train_xgboost_optuna(
    X_train, y_train,
    X_val,   y_val,
    params:   dict,
    n_trials: int = 50,
) -> dict:
    """
    Entraînement XGBoost avec tuning Optuna.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from xgboost import XGBRegressor

    # Espace de recherche depuis model_config.yaml
    search_space = params.get("optuna_search_space", {})
    base_params  = params.get("params", {})

    def objective(trial) -> float:
        p = {
            "n_estimators":     trial.suggest_int(   "n_estimators",     100, 800),
            "learning_rate":    trial.suggest_float(  "learning_rate",    0.005, 0.2,  log=True),
            "max_depth":        trial.suggest_int(   "max_depth",        3, 10),
            "subsample":        trial.suggest_float(  "subsample",        0.5, 1.0),
            "colsample_bytree": trial.suggest_float(  "colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int(   "min_child_weight", 1, 20),
            "gamma":            trial.suggest_float(  "gamma",            0.0, 3.0),
            "reg_alpha":        trial.suggest_float(  "reg_alpha",        1e-8, 5.0, log=True),
            "reg_lambda":       trial.suggest_float(  "reg_lambda",       1e-8, 5.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
        }
        model = XGBRegressor(**p)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return float(np.sqrt(((y_val.values - preds) ** 2).mean()))

    log.info(f"  Optuna XGBoost : {n_trials} trials...")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {**study.best_params, "tree_method": "hist", "random_state": 42, "verbosity": 0}
    log.info(f"  Meilleurs params : {best_params}")

    # Entraînement final sur train + val
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])
    model  = XGBRegressor(**best_params)
    model.fit(X_full, y_full, verbose=False)

    return {"model": model, "best_params": best_params, "model_name": "xgboost"}


# ─── SARIMA ───────────────────────────────────────────────────

def train_sarima(series_train: pd.Series, cfg_model: dict) -> dict:
    """Entraîne SARIMA avec les paramètres de model_config.yaml."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    p = cfg_model["models"]["sarima"]["params"]

    log.info(f"  SARIMA order={p['order']} seasonal={p['seasonal_order']}...")
    model  = SARIMAX(
        series_train,
        order=tuple(p["order"]),
        seasonal_order=tuple(p["seasonal_order"]),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=p.get("maxiter", 200))
    log.info(f"  AIC={result.aic:.2f} | BIC={result.bic:.2f}")

    return {
        "model": result,
        "best_params": {"order": p["order"], "seasonal_order": p["seasonal_order"]},
        "model_name": "sarima",
    }


# ─── Prophet ──────────────────────────────────────────────────

def train_prophet(df_train: pd.DataFrame, df_val: pd.DataFrame,
                  target: str, cfg_model: dict) -> dict:
    """Entraîne Prophet avec régresseurs météo."""
    try:
        from prophet import Prophet
    except ImportError:
        log.warning("Prophet non installé, saut du modèle Prophet")
        return None

    p = cfg_model["models"]["prophet"]["params"]
    regressors = cfg_model["models"]["prophet"].get("add_regressors", [])

    # Préparer les DataFrames Prophet (ds, y, regressors)
    def make_prophet_df(df):
        pdf = df[[target]].reset_index()
        pdf.columns = ["ds", "y"]
        for r in regressors:
            if r in df.columns:
                pdf[r] = df[r].values
        return pdf

    prophet_train = make_prophet_df(df_train)
    prophet_val   = make_prophet_df(df_val)

    model = Prophet(
        yearly_seasonality=p.get("yearly_seasonality", True),
        weekly_seasonality=p.get("weekly_seasonality", True),
        daily_seasonality=p.get("daily_seasonality",  True),
        seasonality_mode=p.get("seasonality_mode", "additive"),
        changepoint_prior_scale=p.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=p.get("seasonality_prior_scale", 10.0),
        interval_width=p.get("interval_width", 0.95),
    )
    for r in regressors:
        if r in prophet_train.columns:
            model.add_regressor(r)

    model.fit(prophet_train)
    log.info("  Prophet entraîné")

    return {
        "model": model,
        "best_params": p,
        "model_name": "prophet",
        "prophet_regressors": regressors,
    }


# ─── Pipeline principal ───────────────────────────────────────

def run_training(
    config_path:       str = "configs/config.yaml",
    model_config_path: str = "configs/model_config.yaml",
    model_override:    str = None,
    n_optuna_trials:   int = 50,
) -> dict:
    """
    Pipeline d'entraînement complet avec MLflow tracking.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(model_config_path) as f:
        cfg_model = yaml.safe_load(f)

    log.info("=" * 55)
    log.info("ENTRAÎNEMENT DES MODÈLES")
    log.info("=" * 55)

    # Chargement
    data = load_datasets(cfg)

    # Sélection du modèle
    from src.models.select import run_model_selection
    best = run_model_selection(
        X_train=data["X_train"], y_train=data["y_train"],
        X_test=data["X_val"],   y_test=data["y_val"],
        model_cfg=cfg_model,
        n_optuna_trials=n_optuna_trials,
    )

    # Évaluation sur le test set
    from src.models.evaluate import compute_metrics
    if best["model_name"] == "xgboost":
        test_preds = best["model"].predict(data["X_test"])
    elif best["model_name"] == "sarima":
        test_preds = best["model"].forecast(steps=len(data["X_test"])).values
    else:
        test_preds = best["model"].predict(data["X_test"])

    test_metrics = compute_metrics(data["y_test"].values, test_preds)

    log.info("\nMétriques TEST SET :")
    for k, v in test_metrics.items():
        log.info(f"  {k.upper():8s} : {v:.4f}")

    # ── MLflow Tracking ──────────────────────────────────────
    try:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name=f"best_{best['model_name']}") as run:
            # Paramètres
            mlflow.log_params(best["best_params"])
            mlflow.log_param("model_name", best["model_name"])
            mlflow.log_param("n_train",    len(data["X_train"]))
            mlflow.log_param("n_features", len(data["feat_cols"]))
            mlflow.log_param("horizon_h",  cfg["project"]["horizon_hours"])

            # Métriques
            mlflow.log_metrics(test_metrics)
            mlflow.log_metrics({f"val_{k}": v for k, v in best["metrics"].items()})

            # Artifacts
            mlflow.log_artifact(config_path)
            mlflow.log_artifact(model_config_path)
            mlflow.log_artifact("params.yaml")

            # Sauvegarder liste des features
            feat_path = "data/processed/feature_names.txt"
            if Path(feat_path).exists():
                mlflow.log_artifact(feat_path)

            # Enregistrement dans le registry
            if best["model_name"] == "xgboost":
                mlflow.sklearn.log_model(
                    sk_model=best["model"],
                    artifact_path="model",
                    registered_model_name=cfg["mlflow"]["model_name"],
                    input_example=data["X_test"].iloc[:3],
                )

            run_id = run.info.run_id
            log.info(f"\nMLflow run : {run_id}")
            log.info(f"   Interface  : {cfg['mlflow']['tracking_uri']}")

    except Exception as e:
        log.warning(f"MLflow tracking échoué (serveur peut-être absent) : {e}")
        run_id = "local_only"

    # ── Sauvegarde locale ────────────────────────────────────
    model_dir = Path(cfg["paths"]["models"]) / "best_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best["model"], model_dir / "model.pkl")

    # Métadonnées du modèle
    metadata = {
        "model_name":    best["model_name"],
        "best_params":   {k: float(v) if isinstance(v, np.floating) else v
                          for k, v in best["best_params"].items()},
        "val_metrics":   {k: float(v) for k, v in best["metrics"].items()},
        "test_metrics":  {k: float(v) for k, v in test_metrics.items()},
        "mlflow_run_id": run_id,
        "n_features":    len(data["feat_cols"]),
        "horizon_h":     cfg["project"]["horizon_hours"],
        "feature_names": data["feat_cols"],
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Métriques pour DVC
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics_train.json", "w") as f:
        json.dump({**test_metrics, "model": best["model_name"]}, f, indent=2)

    log.info(f"\nModèle sauvegardé : {model_dir}")
    log.info(f"   RMSE test : {test_metrics['rmse']:.4f}")
    log.info(f"   MAPE test : {test_metrics['mape']:.2f}%")
    log.info(f"   R²   test : {test_metrics['r2']:.4f}")

    return {**best, "test_metrics": test_metrics, "run_id": run_id}


# ─── CLI ──────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    parser = argparse.ArgumentParser(description="Entraînement du modèle de prévision")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--model",        default=None,
                        help="Forcer un modèle : xgboost, sarima, prophet, lstm")
    parser.add_argument("--trials",       type=int, default=50,
                        help="Nombre de trials Optuna")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        model_config_path=args.model_config,
        model_override=args.model,
        n_optuna_trials=args.trials,
    )
