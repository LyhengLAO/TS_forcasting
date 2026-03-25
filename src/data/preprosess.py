# ============================================================
# src/data/preprocess.py
# Nettoyage, merge, imputation et normalisation des données
#
# Étapes :
#   1. Chargement des données brutes (météo + finance)
#   2. Merge et alignement temporel
#   3. Détection et suppression des outliers (IQR + z-score)
#   4. Imputation temporelle (interpolation + ffill/bfill)
#   5. Sauvegarde en Parquet (interim)
#
# Usage :
#   python -m src.data.preprocess
# ============================================================
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml

log = logging.getLogger(__name__)

# chargement des données brutes (météo + finance)
def load_raw(cfg: dict) -> pd.DataFrame:
    """Charge et merge les données brutes météo + finance."""
    raw_dir      = Path(cfg["paths"]["raw"])
    weather_path = raw_dir / "weather_raw.parquet"
    finance_path = raw_dir / "finance_raw.parquet"

    if not weather_path.exists():
        raise FileNotFoundError(
            f"Données météo manquantes : {weather_path}\n"
            "Lancer d'abord : make collect"
        )

    df_weather = pd.read_parquet(weather_path)
    log.info(f"Météo chargée : {df_weather.shape}")

    # Finance
    if finance_path.exists():
        df_finance = pd.read_parquet(finance_path)
        log.info(f"Finance chargée : {df_finance.shape}")

        # Alignement : reindex finance sur l'index météo (horaire)
        df_finance = df_finance.reindex(df_weather.index, method="ffill", limit=24)

        # Merge left join (on garde TOUTES les heures météo)
        df = df_weather.join(df_finance, how="left")
    else:
        log.warning("Données finance absentes — utilisation météo seule")
        df = df_weather.copy()

    # Tri chronologique strict
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    log.info(f"Dataset merged : {df.shape} | {df.index.min()} → {df.index.max()}")
    return df

# detection des outliers
def remove_outliers_iqr(
    df: pd.DataFrame,
    cols: list,
    factor: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Remplace les outliers par NaN en utilisant la règle IQR.
    factor=3.0 : très conservateur (valeurs à ±3×IQR)
    factor=1.5 : standard (valeurs à ±1.5×IQR)
    """
    df = df.copy()
    stats_dict = {}

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        lower  = Q1 - factor * IQR
        upper  = Q3 + factor * IQR

        mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = mask.sum()
        df.loc[mask, col] = np.nan

        stats_dict[col] = {
            "n_outliers": int(n_outliers),
            "pct_outliers": float(n_outliers / len(df) * 100),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
        }
        if n_outliers > 0:
            log.info(f"  {col:40s} : {n_outliers:5d} outliers ({n_outliers/len(df)*100:.3f}%)")

    return df, stats_dict

def remove_outliers_zscore(
    df: pd.DataFrame,
    cols: list,
    threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Remplace outliers par NaN via z-score > threshold.
    Utilisé en complément de l'IQR pour les queues extrêmes.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        z = np.abs(stats.zscore(df[col].dropna()))
        # Retrouver les indices dans le df original
        non_null_idx = df[col].dropna().index
        extreme_idx  = non_null_idx[z > threshold]
        if len(extreme_idx) > 0:
            df.loc[extreme_idx, col] = np.nan
            log.info(f"  z-score : {col} → {len(extreme_idx)} valeurs extrêmes supprimées")
    return df

# Imputation simple (météo) : forward-fill + backward-fill
def impute_time_series(
    df: pd.DataFrame,
    max_gap_interp: int = 6,    # interpolation sur max 6h consécutives
    max_gap_fill:   int = 24,   # ffill/bfill sur max 24h
) -> pd.DataFrame:
    """
    Imputation adaptée aux séries temporelles :
    1. Interpolation temporelle (linéaire) pour les petits gaps
    2. Forward fill pour les gaps moyens
    3. Backward fill pour les bouts de série
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    # 1. Interpolation temporelle (respecte l'index datetime)
    df[numeric_cols] = df[numeric_cols].interpolate(
        method="time",
        limit=max_gap_interp,
        limit_direction="forward",
    )

    # 2. Forward fill pour les gaps plus longs
    df[numeric_cols] = df[numeric_cols].ffill(limit=max_gap_fill)

    # 3. Backward fill pour les premières heures manquantes
    df[numeric_cols] = df[numeric_cols].bfill(limit=max_gap_fill)

    # Rapport
    total_nulls = df[numeric_cols].isnull().sum().sum()
    if total_nulls > 0:
        null_cols = df[numeric_cols].isnull().sum()
        null_cols = null_cols[null_cols > 0]
        log.warning(f"  {total_nulls} NaN restants après imputation :")
        for col, n in null_cols.items():
            log.warning(f"    {col} : {n}")
    else:
        log.info("Aucun NaN restant après imputation")

    return df
