# ============================================================
# src/models/evaluate.py
# Évaluation complète : métriques, backtesting, plots résidus
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # sans affichage graphique (compatible Windows serveur)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Métriques d'évaluation
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    # Symmetric MAPE
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape, "r2": r2}

# Walk-forward CV pour séries temporelles
def walk_forward_cv(model, X: pd.DataFrame, y: pd.Series,
                    n_splits: int = 5, gap: int = 24) -> dict:
    """
    TimeSeriesSplit avec gap temporel — simule la vraie production.
    À chaque fold : entraîne sur passé, évalue sur futur.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_v = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_v = y.iloc[train_idx], y.iloc[val_idx]

        # Re-entraînement sur chaque fold
        try:
            model.fit(X_tr, y_tr)
        except Exception:
            pass   # certains modèles (SARIMA) ne supportent pas refit ici

        preds = model.predict(X_v)
        m = compute_metrics(y_v.values, preds)
        m["fold"] = fold + 1
        fold_metrics.append(m)
        print(f"  Fold {fold+1}/{n_splits} — RMSE: {m['rmse']:.4f} | R²: {m['r2']:.4f}")

    df = pd.DataFrame(fold_metrics)
    return {
        "cv_rmse_mean": float(df["rmse"].mean()),
        "cv_rmse_std":  float(df["rmse"].std()),
        "cv_mae_mean":  float(df["mae"].mean()),
        "cv_r2_mean":   float(df["r2"].mean()),
        "cv_folds":     fold_metrics,
    }

# Plots
def plot_predictions(y_true: pd.Series, y_pred: np.ndarray, output_dir: str = "reports/figures") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    ax1.plot(y_true.index, y_true.values, label="Réel", color="#185FA5",
             linewidth=1.5, alpha=0.9)
    ax1.plot(y_true.index, y_pred, label="Prédit", color="#D85A30",
             linewidth=1.5, alpha=0.85, linestyle="--")
    ax1.fill_between(y_true.index,
                    y_pred - 2 * 1.5,
                    y_pred + 2 * 1.5,
                    alpha=0.15, color="#D85A30", label="IC 95%")
    ax1.set_title("Températures réelles vs prédites (T+24h)", fontsize=13, fontweight="500")
    ax1.set_ylabel("Température (°C)")
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    n_pts = min(336, len(y_true))   # 2 semaines × 24h × 7j
    ax2.plot(y_true.index[-n_pts:], y_true.values[-n_pts:],
             label="Réel", color="#185FA5", linewidth=2)
    ax2.plot(y_true.index[-n_pts:], y_pred[-n_pts:],
             label="Prédit", color="#D85A30", linewidth=2, linestyle="--")
    ax2.set_title("Zoom — 2 dernières semaines", fontsize=13)
    ax2.set_ylabel("Température (°C)")
    ax2.legend(fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    ax2.grid(True, alpha=0.3)
    #plt.show()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot sauvegardé : {output_dir}/predictions_vs_actual.png")

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   output_dir: str = "reports/figures") -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Distribution des résidus
    axes[0].hist(residuals, bins=50, color="#185FA5", alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="#D85A30", linewidth=2, linestyle="--")
    axes[0].set_title("Distribution des résidus")
    axes[0].set_xlabel("Résidu (°C)")

    # Résidus vs Prédictions (hétéroscédasticité)
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color="#185FA5")
    axes[1].axhline(0, color="#D85A30", linewidth=2, linestyle="--")
    axes[1].set_title("Résidus vs Prédictions")
    axes[1].set_xlabel("Prédit (°C)")
    axes[1].set_ylabel("Résidu (°C)")

    # QQ-plot simplifié
    sorted_res = np.sort(residuals)
    theoretical = np.linspace(-3, 3, len(sorted_res))
    axes[2].scatter(theoretical, sorted_res, alpha=0.3, s=10, color="#185FA5")
    axes[2].plot([-3, 3], [-3 * residuals.std(), 3 * residuals.std()],
                 color="#D85A30", linewidth=2, linestyle="--")
    axes[2].set_title("Q-Q Plot")
    axes[2].set_xlabel("Quantiles théoriques")
    axes[2].set_ylabel("Quantiles observés")
    #plt.show()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, feature_names: list,
                             top_n: int = 25,
                             output_dir: str = "reports/figures") -> None:
    """Compatible XGBoost et sklearn (feature_importances_)."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return   # SARIMA, LSTM → pas de feature importance standard

        idx = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in idx]
        top_values   = importances[idx]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        bars = ax.barh(range(len(top_features)), top_values[::-1],
                       color="#185FA5", alpha=0.85)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1], fontsize=10)
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"Top {top_n} features les plus importantes")
        ax.grid(True, alpha=0.3, axis="x")
        #plt.show()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Feature importance non disponible : {e}")

# Évaluation complète du modèle
def evaluate_model(
    model,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    output_dir: str = "reports/figures",
    run_cv:     bool = False,
) -> dict:
    """
    Évaluation complète du modèle sur le jeu de test.
    Génère métriques + plots.
    """
    print("\nÉvaluation du modèle...")

    # Prédictions
    y_pred = model.predict(X_test)
    m = compute_metrics(y_test.values, y_pred)

    print(f"\n{'─'*40}")
    print(f"  RMSE  : {m['rmse']:.4f}")
    print(f"  MAE   : {m['mae']:.4f}")
    print(f"  MAPE  : {m['mape']:.2f}")
    print(f"  sMAPE : {m['smape']:.2f}")
    print(f"  R²    : {m['r2']:.4f}")
    print(f"{'─'*40}")

    # Walk-forward CV (optionnel, lent)
    if run_cv:
        print("\nWalk-forward cross-validation...")
        cv_results = walk_forward_cv(model, X_test, y_test)
        m.update(cv_results)

    # Plots
    print("\nGénération des plots...")
    plot_predictions(y_test, y_pred, output_dir)
    plot_residuals(y_test.values, y_pred, output_dir)
    plot_feature_importance(model, list(X_test.columns), output_dir=output_dir)

    print("Évaluation terminée")
    return m


if __name__ == "__main__":
    import yaml, joblib
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    TARGET  = cfg["project"]["target"]
    HORIZON = cfg["project"]["horizon_hours"]

    test = pd.read_parquet(
        Path(cfg["paths"]["processed"]) / "features_test.parquet"
    )
    feat_cols = [c for c in test.columns
                 if c != f"target_{HORIZON}h" and c != TARGET]
    model = joblib.load(Path(cfg["paths"]["models"]) / "best_model" / "model.pkl")

    evaluate_model(model, test[feat_cols], test[f"target_{HORIZON}h"])
