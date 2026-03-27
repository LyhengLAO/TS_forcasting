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

# flagging outlier
def flag_outliers(df: pd.DataFrame, outlier_stats: dict) -> pd.DataFrame:
    """
    Ajoute des colonnes binaires indiquant les positions d'outliers.
    Utile pour l'analyse et le monitoring.
    """
    df_flagged = df.copy()
    for col, s in outlier_stats.items():
        if s["n_outliers"] > 0 and col in df.columns:
            flag_col = f"{col}_outlier_flag"
            original = pd.read_parquet(
                "data/raw/weather_raw.parquet"
            )[col] if "meteo_" in col else None
            if original is not None:
                bad = (original < s["lower_bound"]) | (original > s["upper_bound"])
                df_flagged[flag_col] = bad.reindex(df_flagged.index, fill_value=False).astype(int)
    return df_flagged


# Normalisation stats

def compute_normalization_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Calcule la moyenne et l'écart-type des colonnes (sur le train set).
    Sauvegarder pour normaliser les données de production.
    """
    norm_stats = pd.DataFrame({
        "mean": df[cols].mean(),
        "std":  df[cols].std(),
        "min":  df[cols].min(),
        "max":  df[cols].max(),
    })
    out_path = Path("data/interim/normalization_stats.parquet")
    norm_stats.to_parquet(out_path)
    log.info(f"  Stats de normalisation sauvegardées : {out_path}")
    return norm_stats


# pipeline principal

def preprocess(cfg: dict) -> pd.DataFrame:
    """
    Pipeline complet de preprocessing.
    Retourne le DataFrame nettoyé et imputé.
    """
    log.info("=" * 55)
    log.info("PREPROCESSING")
    log.info("=" * 55)

    # 1. Chargement + merge
    df = load_raw(cfg)
    log.info(f"\nShape initial : {df.shape}")

    # 2. Colonnes numériques à traiter
    meteo_cols   = [c for c in df.columns if c.startswith("meteo_")]
    finance_cols = [c for c in df.columns if c.startswith("fin_")]
    all_numeric  = meteo_cols + finance_cols

    # 3. Détection outliers IQR (factor=3 = très conservateur)
    log.info(f"\nDétection outliers IQR (factor=3.0) sur {len(meteo_cols)} colonnes météo :")
    df, outlier_stats = remove_outliers_iqr(df, meteo_cols, factor=3.0)
    total_outliers = sum(s["n_outliers"] for s in outlier_stats.values())
    log.info(f"  Total : {total_outliers} outliers remplacés par NaN")

    # 4. Sauvegarde du dataset avec flags
    flag_path = Path(cfg["paths"]["interim"]) / "outliers_flagged.parquet"
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    df_flagged = df.copy()
    # Ajouter colonnes de flag pour les colonnes ayant eu des outliers
    for col, s in outlier_stats.items():
        if s["n_outliers"] > 0:
            df_flagged[f"{col}_was_outlier"] = 0   # placeholder
    df_flagged.to_parquet(flag_path)
    log.info(f"  Dataset avec flags : {flag_path}")

    # 5. Z-score extrêmes (threshold=5 = vraiment extrêmes)
    log.info(f"\nDétection outliers z-score (threshold=5.0) :")
    df = remove_outliers_zscore(df, meteo_cols, threshold=5.0)

    # 6. Imputation temporelle
    log.info("\nImputation temporelle...")
    df = impute_time_series(df, max_gap_interp=6, max_gap_fill=24)

    # 7. Stats de normalisation (pour le serving)
    log.info("\nCalcul stats de normalisation...")
    compute_normalization_stats(df, meteo_cols)

    # 8. Vérification finale
    remaining_nulls = df[meteo_cols].isnull().sum().sum()
    log.info(f"\nShape final     : {df.shape}")
    log.info(f"Nulls restants  : {remaining_nulls}")
    log.info(f"Types           : {df.dtypes.value_counts().to_dict()}")

    # 9. Sauvegarde
    out_path = Path(cfg["paths"]["interim"]) / "merged_clean.parquet"
    df.to_parquet(out_path, compression="snappy")
    log.info(f"\n Données nettoyées sauvegardées : {out_path}")
    log.info(f"Taille : {out_path.stat().st_size / 1e6:.2f} MB")

    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    preprocess(cfg)
