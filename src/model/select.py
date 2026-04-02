# ============================================================
# src/models/select.py
# Sélection automatique du meilleur modèle via Optuna
# Comparaison : XGBoost, SARIMA, Prophet, LSTM
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


# --- Métriques communes ---

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


# --- XGBoost ---

def _xgb_objective(trial, X_train, y_train, X_val, y_val) -> float:
    from xgboost import XGBRegressor
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma":            trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "tree_method": "hist",
        "random_state": 42,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, preds)))


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _xgb_objective(t, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params
    best_params.update({"tree_method": "hist", "random_state": 42, "verbosity": 0})

    from xgboost import XGBRegressor
    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    return {
        "model": model,
        "best_params": best_params,
        "metrics": metrics(y_val.values, preds),
        "model_name": "xgboost",
    }