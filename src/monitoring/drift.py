# ============================================================
# src/monitoring/drift.py
# Détection de data drift et data quality avec Evidently
#
# Comparaison : distribution de référence (train) vs production (récent)
# Tests : PSI, KL divergence, Kolmogorov-Smirnov, Jensen-Shannon
#
# Usage :
#   python -m src.monitoring.drift
#   python -m src.monitoring.drift --reference-days 30 --current-days 7
# ============================================================

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


# --- Rapport Evidently --- 

def compute_drift_report(
    reference:  pd.DataFrame,
    current:    pd.DataFrame,
    output_dir: str = "reports/monitoring",
    config:     dict = None,
) -> dict:
    """
    Génère un rapport de data drift Evidently.
    Retourne un résumé JSON avec les alertes.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import (
            DataDriftPreset,
            DataQualityPreset,
        )
        from evidently.metrics import (
            DatasetDriftMetric,
            DatasetMissingValuesMetric,
            ColumnDriftMetric,
        )
    except ImportError:
        log.error("Evidently non installé : pip install evidently")
        return _fallback_drift_check(reference, current)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configuration des seuils depuis monitoring_config.yaml
    if config:
        thr = config.get("monitoring", {}).get("drift", {}).get("thresholds", {})
        drift_share_threshold = thr.get("share_drifted_features", 0.20)
    else:
        drift_share_threshold = 0.20

    log.info(f"Drift check : référence={len(reference)} obs | courant={len(current)} obs")

    # Colonnes communes
    common_cols = [c for c in reference.columns
                   if c in current.columns
                   and pd.api.types.is_numeric_dtype(reference[c])]

    if not common_cols:
        log.warning("Aucune colonne commune — drift check ignoré")
        return {"skipped": True, "reason": "no_common_columns"}

    ref_data = reference[common_cols].copy()
    cur_data = current[common_cols].copy()

    # ── Rapport Evidently ──
    report = Report(metrics=[
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    try:
        report.run(reference_data=ref_data, current_data=cur_data)

        # Export HTML interactif
        html_path = f"{output_dir}/drift_report_{ts}.html"
        report.save_html(html_path)
        log.info(f"  📊 Rapport HTML : {html_path}")

        # Extraire résultats
        result_dict = report.as_dict()
        metrics_list = result_dict.get("metrics", [])

        # Trouver DatasetDriftMetric
        drift_metric = None
        for m in metrics_list:
            if "DatasetDriftMetric" in str(m.get("metric", "")):
                drift_metric = m.get("result", {})
                break

        if drift_metric:
            n_drifted = drift_metric.get("number_of_drifted_columns", 0)
            n_total   = drift_metric.get("number_of_columns", len(common_cols))
            share     = drift_metric.get("share_of_drifted_columns", n_drifted / max(n_total, 1))
            dataset_drift = drift_metric.get("dataset_drift", share > drift_share_threshold)
        else:
            # Fallback manuel
            n_drifted    = 0
            n_total      = len(common_cols)
            share        = 0.0
            dataset_drift = False

    except Exception as e:
        log.warning(f"Evidently report échoué : {e}")
        return _fallback_drift_check(ref_data, cur_data, drift_share_threshold)

    # ── Analyse par colonne prioritaire ──
    priority_cols = []
    if config:
        priority_cols = config.get("monitoring", {}).get("drift", {}).get(
            "priority_features", []
        )

    column_drift_details = {}
    for col in (priority_cols or common_cols[:10]):
        if col in ref_data.columns:
            # PSI simplifié
            psi = _compute_psi(ref_data[col].dropna(), cur_data[col].dropna())
            column_drift_details[col] = {
                "psi":    float(psi),
                "drifted": psi > 0.25,
            }

    # ── Résumé ──
    action = "RETRAIN" if share > drift_share_threshold else "MONITOR"

    drift_summary = {
        "timestamp":            ts,
        "dataset_drift":        bool(dataset_drift),
        "n_drifted_features":   int(n_drifted),
        "n_total_features":     int(n_total),
        "share_drifted":        float(share),
        "drift_threshold":      float(drift_share_threshold),
        "action_required":      action,
        "column_details":       column_drift_details,
        "n_reference_samples":  len(reference),
        "n_current_samples":    len(current),
    }

    # Sauvegarder JSON
    json_path = f"{output_dir}/drift_summary_{ts}.json"
    with open(json_path, "w") as fh:
        json.dump(drift_summary, fh, indent=2)

    # Log résumé
    log.info(f"\n{'─'*50}")
    log.info(f"  Dataset drift       : {dataset_drift}")
    log.info(f"  Features en drift   : {n_drifted}/{n_total} ({share:.1%})")
    log.info(f"  Action requise      : {action}")
    log.info(f"  JSON : {json_path}")
    log.info(f"{'─'*50}")

    if action == "RETRAIN":
        log.warning(
            f"⚠️  ALERTE DRIFT : {share:.1%} des features ont drifté "
            f"(seuil={drift_share_threshold:.0%}) → Re-entraînement recommandé"
        )

    return drift_summary


def _compute_psi(
    reference: pd.Series,
    current:   pd.Series,
    n_bins:    int = 10,
) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.10 : stable
    PSI 0.10-0.25 : léger changement
    PSI > 0.25 : drift significatif
    """
    # Créer les bins sur la référence
    _, bins = np.histogram(reference, bins=n_bins)
    bins[0]  = -np.inf
    bins[-1] =  np.inf

    ref_counts = np.histogram(reference, bins=bins)[0]
    cur_counts = np.histogram(current,   bins=bins)[0]

    # Normaliser
    ref_pct = ref_counts / (len(reference) + 1e-10)
    cur_pct = cur_counts / (len(current)   + 1e-10)

    # Éviter division par zéro
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _fallback_drift_check(
    reference:           pd.DataFrame,
    current:             pd.DataFrame,
    drift_threshold:     float = 0.20,
) -> dict:
    """
    Fallback si Evidently n'est pas disponible.
    Utilise PSI et test KS manuellement.
    """
    from scipy import stats

    log.info("  Fallback : drift check via PSI + KS (sans Evidently)")
    common_cols = [c for c in reference.columns if c in current.columns]

    drifted_cols = []
    for col in common_cols[:20]:   # limiter à 20 colonnes
        ref_series = reference[col].dropna()
        cur_series = current[col].dropna()
        if len(ref_series) < 10 or len(cur_series) < 10:
            continue
        ks_stat, ks_p = stats.ks_2samp(ref_series, cur_series)
        psi = _compute_psi(ref_series, cur_series)
        if ks_p < 0.05 or psi > 0.25:
            drifted_cols.append(col)

    share    = len(drifted_cols) / len(common_cols) if common_cols else 0
    action   = "RETRAIN" if share > drift_threshold else "MONITOR"
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "timestamp":          ts,
        "dataset_drift":      share > drift_threshold,
        "n_drifted_features": len(drifted_cols),
        "n_total_features":   len(common_cols),
        "share_drifted":      float(share),
        "action_required":    action,
        "drifted_columns":    drifted_cols,
        "method":             "fallback_psi_ks",
    }


# --- Pipeline monitoring ---

def run_drift_monitoring(
    config_path:           str = "configs/config.yaml",
    monitoring_config_path: str = "configs/monitoring_config.yaml",
    reference_days:        int = 30,
    current_days:          int = 7,
) -> dict:
    """
    Charge les données de référence (train) et les données récentes
    puis calcule le rapport de drift.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(monitoring_config_path) as f:
        mon_cfg = yaml.safe_load(f)

    output_dir = cfg["paths"]["monitoring"]

    # Données de référence : dernières semaines du train set
    train_path = Path(cfg["paths"]["processed"]) / "features_train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Données train manquantes : {train_path}\n"
            "Lancer d'abord : make engineer"
        )

    train = pd.read_parquet(train_path)
    reference = train.tail(reference_days * 24)   # dernières N semaines
    log.info(f"Référence : {len(reference)} obs ({reference.index.min().date()} → {reference.index.max().date()})")

    # Données courantes : jeu de test (simulant la production)
    test_path = Path(cfg["paths"]["processed"]) / "features_test.parquet"
    if test_path.exists():
        current = pd.read_parquet(test_path).tail(current_days * 24)
        log.info(f"Courant   : {len(current)} obs ({current.index.min().date()} → {current.index.max().date()})")
    else:
        log.warning("Test set absent — utilisation des dernières heures du train")
        current = train.tail(current_days * 24)

    # Colonnes à surveiller (exclure target et features engineered)
    base_cols = [
        c for c in reference.columns
        if c.startswith("meteo_") and "lag" not in c
        and "roll" not in c and "diff" not in c
        and "pct" not in c and "ewm" not in c
    ]

    ref_data = reference[base_cols]
    cur_data = current[base_cols] if all(c in current.columns for c in base_cols) else current

    return compute_drift_report(
        reference=ref_data,
        current=cur_data,
        output_dir=output_dir,
        config=mon_cfg,
    )


# --- CLI ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    parser = argparse.ArgumentParser(description="Monitoring drift des données")
    parser.add_argument("--config",            default="configs/config.yaml")
    parser.add_argument("--monitoring-config", default="configs/monitoring_config.yaml")
    parser.add_argument("--reference-days",    type=int, default=30)
    parser.add_argument("--current-days",      type=int, default=7)
    args = parser.parse_args()

    result = run_drift_monitoring(
        config_path=args.config,
        monitoring_config_path=args.monitoring_config,
        reference_days=args.reference_days,
        current_days=args.current_days,
    )
    print(f"\nAction requise : {result.get('action_required', 'N/A')}")
