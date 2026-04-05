# ============================================================
# src/monitoring/performance.py
# Suivi de la performance du modèle en production :
#   - RMSE, MAE, MAPE glissants
#   - Détection de dégradation par rapport au baseline
#   - Décision de re-entraînement
#   - Export métriques Prometheus
#
# Usage :
#   python -m src.monitoring.performance
# ============================================================

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


# ─── Métriques glissantes ─────────────────────────────────────

def compute_rolling_metrics(
    predictions:  pd.DataFrame,
    window_hours: int = 72,
) -> dict:
    """
    Calcule les métriques sur une fenêtre glissante.

    predictions doit avoir les colonnes :
      - predicted  : valeur prédite
      - actual     : valeur réelle (ground truth, disponible avec délai)
      - timestamp  : index datetime
    """
    if predictions.empty:
        return {}

    # Garder uniquement les lignes avec ground truth disponible
    df = predictions.dropna(subset=["actual", "predicted"]).copy()
    if len(df) == 0:
        log.warning("Aucune prédiction avec ground truth disponible")
        return {}

    # Fenêtre glissante
    cutoff = df.index.max() - pd.Timedelta(hours=window_hours)
    df_window = df[df.index >= cutoff]

    if len(df_window) < 10:
        log.warning(f"Trop peu de données dans la fenêtre ({len(df_window)} obs)")
        return {}

    y_true = df_window["actual"].values
    y_pred = df_window["predicted"].values

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    bias = float(np.mean(y_pred - y_true))   # biais systématique

    return {
        "window_hours":  window_hours,
        "n_samples":     len(df_window),
        "rmse":          rmse,
        "mae":           mae,
        "mape":          mape,
        "bias":          bias,
        "window_start":  str(df_window.index.min()),
        "window_end":    str(df_window.index.max()),
    }


# ─── Comparaison avec baseline ────────────────────────────────

def check_performance_degradation(
    current_metrics: dict,
    baseline_metrics: dict,
    thresholds: dict,
) -> dict:
    """
    Compare les métriques courantes avec le baseline de validation.
    Détecte une dégradation si les seuils sont dépassés.
    """
    alerts = {}

    if not current_metrics or not baseline_metrics:
        return {"no_data": True}

    # RMSE : dégradation relative
    if "rmse" in current_metrics and "rmse" in baseline_metrics:
        baseline_rmse  = baseline_metrics["rmse"]
        current_rmse   = current_metrics["rmse"]
        relative_increase = (current_rmse - baseline_rmse) / (baseline_rmse + 1e-8)
        threshold = thresholds.get("rmse_relative_increase", 0.30)

        alerts["rmse_degradation"] = {
            "baseline":          float(baseline_rmse),
            "current":           float(current_rmse),
            "relative_increase": float(relative_increase),
            "threshold":         float(threshold),
            "alert":             bool(relative_increase > threshold),
        }
        if relative_increase > threshold:
            log.warning(
                f"⚠️  DÉGRADATION RMSE : {current_rmse:.4f} vs baseline {baseline_rmse:.4f} "
                f"(+{relative_increase:.1%}, seuil={threshold:.0%})"
            )

    # MAPE : dégradation absolue
    if "mape" in current_metrics and "mape" in baseline_metrics:
        baseline_mape  = baseline_metrics["mape"]
        current_mape   = current_metrics["mape"]
        absolute_increase = current_mape - baseline_mape
        threshold = thresholds.get("mape_absolute_increase", 5.0)

        alerts["mape_degradation"] = {
            "baseline":          float(baseline_mape),
            "current":           float(current_mape),
            "absolute_increase": float(absolute_increase),
            "threshold":         float(threshold),
            "alert":             bool(absolute_increase > threshold),
        }

    # Biais systématique
    if "bias" in current_metrics:
        bias = abs(current_metrics["bias"])
        alerts["systematic_bias"] = {
            "bias":  float(bias),
            "alert": bool(bias > 2.0),   # >2°C de biais systématique
        }

    # Décision globale
    n_alerts = sum(
        1 for a in alerts.values()
        if isinstance(a, dict) and a.get("alert", False)
    )
    alerts["n_alerts"]    = n_alerts
    alerts["action"]      = "RETRAIN" if n_alerts >= 2 else (
                             "MONITOR" if n_alerts == 1 else "OK"
                            )
    alerts["timestamp"]   = datetime.now().isoformat()

    return alerts


# ─── Chargement du baseline ───────────────────────────────────

def load_baseline_metrics(metrics_path: str = "reports/metrics_test.json") -> dict:
    """Charge les métriques de validation initiales du modèle."""
    p = Path(metrics_path)
    if not p.exists():
        log.warning(f"Métriques baseline manquantes : {p}")
        return {}
    with open(p) as f:
        return json.load(f)


# ─── Simulation de prédictions production ─────────────────────

def load_production_predictions(
    predictions_dir: str = "reports/",
    lookback_hours:  int = 168,
) -> pd.DataFrame:
    """
    Charge les prédictions sauvegardées par le pipeline d'inférence.
    En production réelle, ces données viendraient d'une base de données.
    """
    pred_dir = Path(predictions_dir)
    pred_files = sorted(pred_dir.glob("predictions_*.parquet"))

    if not pred_files:
        log.warning("Aucun fichier de prédictions trouvé")
        return pd.DataFrame()

    frames = []
    for f in pred_files[-7:]:   # 7 derniers fichiers max
        try:
            df = pd.read_parquet(f)
            frames.append(df)
        except Exception as e:
            log.warning(f"  Impossible de lire {f} : {e}")

    if not frames:
        return pd.DataFrame()

    predictions = pd.concat(frames).sort_index()
    predictions = predictions[~predictions.index.duplicated(keep="last")]

    # Simuler le ground truth (en production : récupéré depuis l'API après délai)
    # Ici on utilise les données test comme proxy
    test_path = Path("data/processed/features_test.parquet")
    if test_path.exists():
        test_data   = pd.read_parquet(test_path)
        target_col  = [c for c in test_data.columns if c.startswith("target_")]
        actual_col  = target_col[0] if target_col else "meteo_temperature_2m"

        # Aligner sur l'index des prédictions
        actual = test_data[actual_col].reindex(predictions.index, method="nearest")
        predictions["actual"] = actual

    return predictions


# ─── Export métriques Prometheus ──────────────────────────────

def export_prometheus_metrics(
    rolling_metrics: dict,
    degradation:     dict,
    output_path:     str = "reports/monitoring/prometheus_metrics.prom",
) -> None:
    """
    Exporte les métriques au format Prometheus text format.
    Peut être consommé par un node exporter ou pushgateway.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HELP model_rmse_live RMSE calculé sur la fenêtre glissante",
        "# TYPE model_rmse_live gauge",
        f"model_rmse_live {rolling_metrics.get('rmse', 0):.6f}",
        "",
        "# HELP model_mae_live MAE calculé sur la fenêtre glissante",
        "# TYPE model_mae_live gauge",
        f"model_mae_live {rolling_metrics.get('mae', 0):.6f}",
        "",
        "# HELP model_mape_live MAPE calculé sur la fenêtre glissante",
        "# TYPE model_mape_live gauge",
        f"model_mape_live {rolling_metrics.get('mape', 0):.6f}",
        "",
        "# HELP model_bias_live Biais systématique (pred - actual)",
        "# TYPE model_bias_live gauge",
        f"model_bias_live {rolling_metrics.get('bias', 0):.6f}",
        "",
        "# HELP performance_alerts_total Nombre d'alertes de dégradation",
        "# TYPE performance_alerts_total gauge",
        f"performance_alerts_total {degradation.get('n_alerts', 0)}",
        "",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"  Métriques Prometheus exportées : {output_path}")


# ─── Pipeline principal ───────────────────────────────────────

def run_performance_monitoring(
    config_path:           str = "configs/config.yaml",
    monitoring_config_path: str = "configs/monitoring_config.yaml",
) -> dict:
    """
    Pipeline complet de monitoring des performances.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(monitoring_config_path) as f:
        mon_cfg = yaml.safe_load(f)

    perf_cfg    = mon_cfg.get("monitoring", {}).get("performance", {})
    thresholds  = perf_cfg.get("degradation_thresholds", {})
    window_h    = perf_cfg.get("rolling_window_hours", 72)
    output_dir  = cfg["paths"]["monitoring"]

    log.info("=" * 55)
    log.info("MONITORING PERFORMANCE")
    log.info("=" * 55)

    # 1. Charger prédictions production
    log.info("\n▶ Chargement prédictions production...")
    predictions = load_production_predictions()

    if predictions.empty:
        log.warning("Aucune prédiction disponible — monitoring ignoré")
        return {"status": "no_predictions"}

    # 2. Métriques glissantes
    log.info(f"\n▶ Métriques glissantes ({window_h}h)...")
    rolling = compute_rolling_metrics(predictions, window_hours=window_h)
    if rolling:
        log.info(f"  RMSE  : {rolling.get('rmse', 0):.4f}")
        log.info(f"  MAE   : {rolling.get('mae', 0):.4f}")
        log.info(f"  MAPE  : {rolling.get('mape', 0):.2f}%")
        log.info(f"  Biais : {rolling.get('bias', 0):.4f}")

    # 3. Comparaison avec baseline
    log.info("\n▶ Comparaison avec baseline...")
    baseline = load_baseline_metrics()
    degradation = check_performance_degradation(rolling, baseline, thresholds)

    # 4. Export
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = {
        "timestamp":    ts,
        "rolling":      rolling,
        "degradation":  degradation,
        "action":       degradation.get("action", "MONITOR"),
    }

    with open(f"{output_dir}/performance_{ts}.json", "w") as f:
        json.dump(result, f, indent=2)

    # 5. Prometheus
    export_prometheus_metrics(rolling, degradation, f"{output_dir}/prometheus_metrics.prom")

    log.info(f"\n{'─'*50}")
    log.info(f"  Action recommandée : {result['action']}")
    log.info(f"  Alertes actives    : {degradation.get('n_alerts', 0)}")
    log.info(f"{'─'*50}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")
    run_performance_monitoring()
