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

# --- SARIMA ---
def tune_sarima(series_train: pd.Series, series_val: pd.Series,
                model_cfg: dict) -> dict:
    """
    Grille réduite (auto_arima via pmdarima ou config fixe).
    SARIMA est lent — on utilise les paramètres de model_config.yaml
    sauf si pmdarima est installé.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cfg = model_cfg["models"]["sarima"]["params"]
    order          = tuple(cfg["order"])
    seasonal_order = tuple(cfg["seasonal_order"])

    model  = SARIMAX(series_train, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=cfg.get("maxiter", 200))

    forecast = result.forecast(steps=len(series_val))
    m = metrics(series_val.values, forecast.values)

    return {
        "model": result,
        "best_params": {"order": order, "seasonal_order": seasonal_order},
        "metrics": m,
        "model_name": "sarima",
    }

# --- LSTM ---
def tune_lstm(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model_cfg: dict, input_size: int) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    cfg = model_cfg["models"]["lstm"]
    arch = cfg["architecture"]
    tr   = cfg["training"]

    hidden_size = int(arch["hidden_size"])
    num_layers  = int(arch["num_layers"])
    dropout     = float(arch["dropout"])
    seq_len     = int(tr["sequence_length"])
    epochs      = int(tr["epochs"])
    batch_size  = int(tr["batch_size"])
    lr          = float(tr["learning_rate"])
    patience    = int(tr["patience"])
    clip_grad   = float(tr["clip_grad_norm"])
    weight_decay= float(tr["weight_decay"])
    T_max       = int(cfg["scheduler"]["T_max"])  # ← probable source du bug

    # Préparer séquences
    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return np.array(Xs), np.array(ys)

    X_tr_s, y_tr_s = make_sequences(X_train, y_train, seq_len)
    X_v_s,  y_v_s  = make_sequences(X_val,   y_val,   seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_tr_s).to(device),
        torch.FloatTensor(y_tr_s).to(device),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_v_s).to(device),
        torch.FloatTensor(y_v_s).to(device),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Modèle
    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTMForecaster().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=tr["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tr["scheduler"]["T_max"]
    )

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tr["clip_grad_norm"])
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_dl:
                val_losses.append(criterion(model(Xb), yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # Charger meilleur état
    if best_state:
        model.load_state_dict(best_state)

    # Évaluer
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in val_dl:
            preds.extend(model(Xb).cpu().numpy())
    m = metrics(y_v_s, np.array(preds))

    return {
        "model": model,
        "best_params": {
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "lr": lr, "seq_len": seq_len,
        },
        "metrics": m,
        "model_name": "lstm",
    }

# --- Prophet ---

def tune_prophet(df_train: pd.DataFrame, df_val: pd.DataFrame,
                 target: str, model_cfg: dict) -> dict:
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet non installé : pip install prophet")

    cfg        = model_cfg["models"]["prophet"]
    p          = cfg["params"]
    regressors = cfg.get("add_regressors", [])

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
        changepoint_prior_scale=float(p.get("changepoint_prior_scale", 0.05)),
        seasonality_prior_scale=float(p.get("seasonality_prior_scale", 10.0)),
        interval_width=float(p.get("interval_width", 0.95)),
    )
    for r in regressors:
        if r in prophet_train.columns:
            model.add_regressor(r)

    model.fit(prophet_train)

    # Prédiction sur val
    future = prophet_val[["ds"] + [r for r in regressors if r in prophet_val.columns]]
    forecast = model.predict(future)
    preds = forecast["yhat"].values

    m = metrics(prophet_val["y"].values, preds)

    return {
        "model":       model,
        "best_params": p,
        "metrics":     m,
        "model_name":  "prophet",
    }

# --- selection finale ---
def run_model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    model_cfg: dict,
    n_optuna_trials: int = 50,
) -> dict:
    """
    Compare tous les modèles activés dans model_config.yaml,
    retourne le meilleur selon RMSE sur le jeu de validation.
    """
    # Split train → train + val (80/20 temporel)
    n = len(X_train)
    split = int(n * 0.8)
    X_tr, y_tr = X_train.iloc[:split],  y_train.iloc[:split]
    X_vl, y_vl = X_train.iloc[split:],  y_train.iloc[split:]

    results = {}
    enabled_models = {
        k: v for k, v in model_cfg["models"].items() if v.get("enabled", False)
    }
    # Trier par priorité
    ordered = sorted(enabled_models.items(), key=lambda x: x[1].get("priority", 99))

    for model_name, _ in ordered:
        print(f"\n{'='*50}")
        print(f"▶ Tuning : {model_name.upper()}")
        print(f"{'='*50}")

        try:
            if model_name == "xgboost":
                res = tune_xgboost(X_tr, y_tr, X_vl, y_vl, n_trials=n_optuna_trials)

            elif model_name == "sarima":
                target_col = y_tr.name if hasattr(y_tr, "name") else "target"
                res = tune_sarima(y_tr, y_vl, model_cfg)

            elif model_name == "lstm":
                res = tune_lstm(
                    X_tr.values, y_tr.values,
                    X_vl.values, y_vl.values,
                    model_cfg, input_size=X_tr.shape[1],
                )

            elif model_name == "prophet":
                res = tune_prophet(
                    df_train=X_tr.assign(**{y_tr.name: y_tr}),
                    df_val=X_vl.assign(**{y_vl.name: y_vl}),
                    target=y_tr.name,
                    model_cfg=model_cfg,
                )
            else:
                print(f"Modèle inconnu : {model_name} — ignoré")
                continue

            results[model_name] = res
            m = res["metrics"]
            print(f"  RMSE={m['rmse']:.4f} | MAE={m['mae']:.4f} "
                  f"| MAPE={m['mape']:.2f}% | R²={m['r2']:.4f}")

        except Exception as e:
            print(f"Erreur {model_name} : {e}")
            continue

    if not results:
        raise RuntimeError("Aucun modèle entraîné avec succès")

    # Sélection du meilleur par RMSE
    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    best = results[best_name]

    print(f"\n{'='*50}")
    print(f"MEILLEUR MODÈLE : {best_name.upper()}")
    print(f"RMSE = {best['metrics']['rmse']:.4f}")
    print(f"{'='*50}")

    return best
