# ============================================================
# src/data/validate.py
# Validation des données brutes avec pandas / numpy natifs
# Vérifie : schéma, types, plages physiques, complétude, continuité
#
# Usage :
#   python -m src.data.validate
# ============================================================
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# regles de validation pour les données météo
WEATHER_RULES = {
    # Colonne requise : min, max, null_ratio_max
    "meteo_temperature_2m":       {"min": -60.0,  "max": 60.0,   "max_null": 0.02},
    "meteo_relative_humidity_2m": {"min":   0.0,  "max": 100.0,  "max_null": 0.02},
    "meteo_wind_speed_10m":       {"min":   0.0,  "max": 250.0,  "max_null": 0.05},
    "meteo_precipitation":        {"min":   0.0,  "max": 500.0,  "max_null": 0.05},
    "meteo_surface_pressure":     {"min": 870.0,  "max": 1085.0, "max_null": 0.05},
    "meteo_shortwave_radiation":  {"min":   0.0,  "max": 1400.0, "max_null": 0.10},
}

NUMERIC_KINDS = {"f", "i", "u"}

def _check(results: dict, key: str, passed: bool) -> bool:
    results[key] = passed
    return passed

def validate_weather(df: pd.DataFrame) -> dict[str, bool]:
    """
    Valide le DataFrame météo sans Great Expectations.
    Retourne un dict {check_name: passed}.
    Lève ValueError si un check critique échoue.
    """
    log.info("🔍  Validation des données météo...")
    results: dict[str, bool] = {}
    errors:  list[str]       = []

    # 1. Colonnes requises présentes et de type numérique
    for col in WEATHER_RULES:
        passed = col in df.columns
        _check(results, f"col_exists_{col}", passed)
        if not passed:
            errors.append(f"Colonne manquante : {col}")
        
        if col not in df.columns:
            continue
        passed = df[col].dtype.kind in NUMERIC_KINDS
        _check(results, f"type_numeric_{col}", passed)

    # 2. Valeurs dans les plages physiques plausibles
    for col, rules in WEATHER_RULES.items():
        if col not in df.columns:
            continue
        series      = df[col].dropna()
        in_range    = series.between(rules["min"], rules["max"])
        pct_ok      = in_range.mean() if len(series) else 0.0
        passed      = pct_ok >= 0.99
        _check(results, f"range_{col}", passed)
        if not passed:
            pct_bad = (1 - pct_ok) * 100
            log.warning(f"  {col} : {pct_bad:.2f}% de valeurs hors plage")
        
        null_ratio = df[col].isnull().mean()
        passed     = null_ratio <= rules["max_null"]
        _check(results, f"nulls_{col}", passed)
        if not passed:
            log.warning(
                f"  {col} : {null_ratio * 100:.2f}% nulls "
                f"(max={rules['max_null'] * 100:.0f}%)"
            )

    # 3. Index temporel unique et ordonné
    dup_mask = df.index.duplicated()
    _check(results, "index_unique", not dup_mask.any())
    _check(results, "index_monotonic", df.index.is_monotonic_increasing)
    _check(results, "index_is_datetime", isinstance(df.index, pd.DatetimeIndex))

    if dup_mask.any():
        errors.append(f"Index non-unique : {dup_mask.sum()} doublons")
    if not df.index.is_monotonic_increasing:
        errors.append("Index non-ordonné chronologiquement")

    # 4. Couverture temporelle d'au moins 1 an et continuité (gap max 24h)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
        duration_days = (df.index.max() - df.index.min()).days
        passed        = duration_days >= 365
        _check(results, "min_coverage_1year", passed)
        if not passed:
            log.warning(f"  Couverture temporelle insuffisante : {duration_days} jours")
        
        diffs     = df.index.to_series().diff().dt.total_seconds().dropna() / 3600
        max_gap_h = diffs.max()
        passed    = max_gap_h <= 24
        _check(results, "max_gap_24h", passed)
        if not passed:
            log.warning(f"  Gap maximal : {max_gap_h:.0f}h (seuil=24h)")
        
    # ── Résumé ──
    n_passed = sum(results.values())
    n_total  = len(results)
    log.info(f"  {n_passed}/{n_total} checks passés")

    # Sauvegarde du rapport
    Path("reports").mkdir(exist_ok=True)
    with open("reports/validation_report.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Lever une erreur si des checks critiques ont échoué
    critical_failed = [
        k for k, v in results.items()
        if not v and any(x in k for x in ("col_exists", "index_unique", "index_monotonic"))
    ]
    if critical_failed:
        raise ValueError(
            f"Validation échouée sur checks critiques : {critical_failed}\n"
            f"Erreurs : {errors}"
        )

    log.info("Validation météo réussie")
    return results

# Validation finance
def validate_finance(df: pd.DataFrame) -> dict[str, bool]:
    """Validation basique des données financières."""
    if df.empty:
        return {"data_not_empty": False}

    results: dict[str, bool] = {}

    # Colonnes OHLCV par ticker
    close_cols = [c for c in df.columns if c.endswith("_close")]
    for col in close_cols:
        series = df[col].dropna()

        # 99 % des valeurs strictement positives
        pct_positive = (series > 0).mean() if len(series) else 0.0
        _check(results, f"close_positive_{col}", pct_positive >= 0.99)

        # Moins de 20 % de nulls
        null_ratio = df[col].isnull().mean()
        _check(results, f"close_no_nulls_{col}", null_ratio < 0.20)

    _check(results, "index_unique",    not df.index.duplicated().any())
    _check(results, "index_monotonic", df.index.is_monotonic_increasing)

    log.info(
        f"Validation finance : "
        f"{sum(results.values())}/{len(results)} checks passés"
    )
    return results

# pipeline de validation complète
def run_validation(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir     = Path(cfg["paths"]["raw"])
    all_results = {}

    # Valider météo
    weather_path = raw_dir / "weather_raw.parquet"
    if weather_path.exists():
        df_weather = pd.read_parquet(weather_path)
        all_results["weather"] = validate_weather(df_weather)
    else:
        raise FileNotFoundError(
            f"Données météo brutes introuvables : {weather_path}\n"
            "Lancer d'abord : make collect"
        )

    # Valider finance (optionnel)
    finance_path = raw_dir / "finance_raw.parquet"
    if finance_path.exists():
        df_finance = pd.read_parquet(finance_path)
        all_results["finance"] = validate_finance(df_finance)
    else:
        log.warning("Données finance absentes — validation ignorée")

    return all_results

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    run_validation()