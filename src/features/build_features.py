# ============================================================
# src/features/build_features.py
# Feature engineering complet pour séries temporelles météo
#
# Features générées :
#   - Encodages cycliques (sin/cos) : heure, jour semaine, mois
#   - Lags temporels : T-1h à T-168h
#   - Statistiques glissantes : mean, std, min, max, range
#   - Différenciation : rend la série stationnaire
#   - Features calendaires : week-end, saison, mois
#   - Features météo croisées : humidex, windchill, heat index
#
# Usage :
#   python -m src.features.build_features
# ============================================================
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

TARGET = "meteo_temperature_2m"


# features cycliques pour les variables temporelles
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodage cyclique via sin/cos — garantit la continuité
    (pas de rupture entre 23h et 0h, entre déc et jan, etc.)
    """
    df = df.copy()
    idx = df.index

    # Heure (cycle 24h)
    df["hour_sin"]   = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * idx.hour / 24)

    # Jour de la semaine (cycle 7j)
    df["dow_sin"]    = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * idx.dayofweek / 7)

    # Mois (cycle 12m)
    df["month_sin"]  = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * idx.month / 12)

    # Jour de l'année (cycle 365j) — saisonnalité annuelle fine
    df["doy_sin"]    = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["doy_cos"]    = np.cos(2 * np.pi * idx.dayofyear / 365.25)

    # Features calendaires binaires
    df["is_weekend"]  = (idx.dayofweek >= 5).astype(np.int8)
    df["is_daytime"]  = ((idx.hour >= 6) & (idx.hour < 20)).astype(np.int8)
    df["is_morning"]  = ((idx.hour >= 6) & (idx.hour < 12)).astype(np.int8)
    df["is_afternoon"]= ((idx.hour >= 12) & (idx.hour < 18)).astype(np.int8)

    # Saison (hémisphère nord)
    seasons = {12: 0, 1: 0, 2: 0,   # Hiver
               3: 1, 4: 1, 5: 1,    # Printemps
               6: 2, 7: 2, 8: 2,    # Été
               9: 3, 10: 3, 11: 3}  # Automne
    df["season"] = idx.month.map(seasons).astype(np.int8)

    return df

# Lags temporels pour capturer les dépendances passées
def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list,
) -> pd.DataFrame:
    """
    Lags temporels : valeurs passées comme features prédictives.
    Indispensables pour les modèles non-récurrents (XGBoost).
    """
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df

# statistiques glissantes pour capturer les tendances locales
def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list = [6, 24, 168],
) -> pd.DataFrame:
    """
    Statistiques glissantes : capturent les tendances locales.
    windows = [6h, 24h, 168h=1semaine]
    """
    df = df.copy()
    for w in windows:
        label = f"{w}h" if w < 168 else "7d"
        series = df[col]

        df[f"{col}_roll_mean_{label}"]  = series.rolling(w, min_periods=max(1, w // 4)).mean()
        df[f"{col}_roll_std_{label}"]   = series.rolling(w, min_periods=max(1, w // 4)).std()
        df[f"{col}_roll_min_{label}"]   = series.rolling(w, min_periods=max(1, w // 4)).min()
        df[f"{col}_roll_max_{label}"]   = series.rolling(w, min_periods=max(1, w // 4)).max()
        df[f"{col}_roll_range_{label}"] = (
            df[f"{col}_roll_max_{label}"] - df[f"{col}_roll_min_{label}"]
        )

        # Exponentially Weighted Mean (réactivité aux changements récents)
        df[f"{col}_ewm_{label}"] = series.ewm(span=w, min_periods=1).mean()

    return df

# différenciation pour rendre la série stationnaire (capturer les changements plutôt que les niveaux)
def add_differencing(
    df: pd.DataFrame,
    col: str,
    diffs: list = [1, 24, 168],
) -> pd.DataFrame:
    """
    Différenciation : capte les variations relatives.
    diff(1)   = changement heure sur heure
    diff(24)  = changement jour sur jour (même heure)
    diff(168) = changement semaine sur semaine
    """
    df = df.copy()
    for d in diffs:
        label = f"{d}h" if d < 168 else "7d"
        df[f"{col}_diff_{label}"] = df[col].diff(d)
        # Taux de variation en %
        df[f"{col}_pct_{label}"] = df[col].pct_change(d).clip(-2, 2)
    return df

# features météo dérivées pour capturer les interactions physiques
def add_derived_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indices météo physiquement significatifs :
    - Humidex : ressenti humidité + chaleur
    - Windchill : ressenti froid + vent
    - Radiation nette approchée
    """
    df = df.copy()

    if "meteo_temperature_2m" in df.columns and "meteo_relative_humidity_2m" in df.columns:
        T  = df["meteo_temperature_2m"]
        RH = df["meteo_relative_humidity_2m"]
        W  = df.get("meteo_wind_speed_10m", pd.Series(0, index=df.index))

        # Pression de vapeur saturante (formule de Magnus)
        e  = 6.112 * np.exp(17.67 * T / (T + 243.5))
        es = RH / 100 * e

        # Humidex (Masterton & Richardson, 1979)
        df["meteo_humidex"] = T + 0.5555 * (es - 10)

        # Windchill (formule de Siple & Passel simplifiée, valide si T < 10°C et W > 1.3 m/s)
        W_kmh = W * 3.6
        windchill = (
            13.12 + 0.6215 * T
            - 11.37 * (W_kmh ** 0.16)
            + 0.3965 * T * (W_kmh ** 0.16)
        )
        # Appliquer seulement si conditions valides
        valid_wc = (T < 10) & (W > 1.3)
        df["meteo_windchill"] = np.where(valid_wc, windchill, T)

        # Température ressentie globale
        df["meteo_apparent_temp"] = np.where(
            T > 27, df["meteo_humidex"],
            np.where(T < 10, df["meteo_windchill"], T)
        )

    # Radiation solaire normalisée (0-1)
    if "meteo_shortwave_radiation" in df.columns:
        max_radiation = df["meteo_shortwave_radiation"].quantile(0.999)
        df["meteo_radiation_norm"] = (
            df["meteo_shortwave_radiation"] / (max_radiation + 1e-8)
        ).clip(0, 1)

    return df

# interactions entre features pour capturer les non-linéarités physiques
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactions multiplicatives entre variables météo clés.
    Capturent des effets non-linéaires importants.
    """
    df = df.copy()

    if all(c in df.columns for c in ["meteo_temperature_2m", "meteo_relative_humidity_2m"]):
        df["interaction_temp_x_hum"] = (
            df["meteo_temperature_2m"] * df["meteo_relative_humidity_2m"] / 100
        )

    if all(c in df.columns for c in ["meteo_temperature_2m", "meteo_wind_speed_10m"]):
        df["interaction_temp_x_wind"] = (
            df["meteo_temperature_2m"] * df["meteo_wind_speed_10m"]
        )

    if all(c in df.columns for c in ["meteo_shortwave_radiation", "meteo_cloudcover"]):
        df["interaction_rad_x_cloud"] = (
            df["meteo_shortwave_radiation"] * (1 - df["meteo_cloudcover"] / 100)
        )

    return df

# split temporel pour éviter les fuites de données
def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologique strict — JAMAIS de shuffle.
    Train | Val | Test dans l'ordre temporel.
    """
    n     = len(df)
    i_tr  = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))

    train = df.iloc[:i_tr]
    val   = df.iloc[i_tr:i_val]
    test  = df.iloc[i_val:]

    log.info(f"  Train : {train.shape} | {train.index.min().date()} → {train.index.max().date()}")
    log.info(f"  Val   : {val.shape}   | {val.index.min().date()} → {val.index.max().date()}")
    log.info(f"  Test  : {test.shape}  | {test.index.min().date()} → {test.index.max().date()}")

    return train, val, test

# pipeline principal
def build_features(
    cfg: dict,
    horizon_h: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline complet de feature engineering.
    Retourne (train, test) avec la variable cible target_{horizon}h.
    """
    log.info("=" * 55)
    log.info("FEATURE ENGINEERING")
    log.info("=" * 55)

    interim_path = Path(cfg["paths"]["interim"]) / "merged_clean.parquet"
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Données nettoyées manquantes : {interim_path}\n"
            "Lancer d'abord : make engineer"
        )

    df = pd.read_parquet(interim_path)
    log.info(f"Données chargées : {df.shape}")

    # Récupération de la config features
    feat_cfg = cfg.get("feature_engineering", {})
    lags     = feat_cfg.get("lags", [1, 2, 3, 6, 12, 24, 48, 72, 168])
    windows  = feat_cfg.get("rolling_windows", [6, 24, 168])
    diffs    = feat_cfg.get("diff_orders", [1, 24, 168])

    # 1. Features temporelles cycliques
    log.info("\nFeatures temporelles...")
    df = add_time_features(df)

    # 2. Lags sur la cible
    log.info(f"Lags {lags}...")
    df = add_lag_features(df, TARGET, lags=lags)

    # 3. Lags sur autres variables météo importantes
    for col in ["meteo_relative_humidity_2m", "meteo_wind_speed_10m",
                "meteo_shortwave_radiation"]:
        if col in df.columns:
            df = add_lag_features(df, col, lags=[1, 6, 24])

    # 4. Rolling stats
    log.info(f"Rolling stats {windows}...")
    df = add_rolling_features(df, TARGET, windows=windows)

    # 5. Différenciation
    log.info(f"Différenciation {diffs}...")
    df = add_differencing(df, TARGET, diffs=diffs)

    # 6. Features météo dérivées
    log.info("Features météo dérivées (humidex, windchill)...")
    df = add_derived_weather_features(df)

    # 7. Interactions
    log.info("Features d'interaction...")
    df = add_interaction_features(df)

    # 8. Variable cible : T+horizon
    target_col = f"target_{horizon_h}h"
    df[target_col] = df[TARGET].shift(-horizon_h)

    # 9. Supprimer les NaN (créés par lags + target shift)
    n_before = len(df)
    df = df.dropna()
    n_after  = len(df)
    log.info(f"\nNaN supprimés : {n_before - n_after} ({(n_before - n_after)/n_before*100:.1f}%)")

    # 10. Infos sur les features générées
    feat_cols = [c for c in df.columns if c != target_col and c != TARGET]
    log.info(f"Features totales : {len(feat_cols)}")
    log.info(f"Shape finale     : {df.shape}")

    # 11. Split temporel
    log.info("\nSplit temporel...")
    train_ratio = cfg.get("data", {}).get("train_ratio", 0.80)
    val_ratio   = cfg.get("data", {}).get("val_ratio",   0.10)
    train, val, test = temporal_split(df, train_ratio, val_ratio)

    # 12. Sauvegarde
    out_dir = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out_dir / "features_train.parquet", compression="snappy")
    val.to_parquet(  out_dir / "features_val.parquet",   compression="snappy")
    test.to_parquet( out_dir / "features_test.parquet",  compression="snappy")

    log.info(f"\nFeatures sauvegardées dans {out_dir}")
    log.info(f"   Train : {len(train)} obs")
    log.info(f"   Val   : {len(val)} obs")
    log.info(f"   Test  : {len(test)} obs")

    # Sauvegarde de la liste des features (pour l'API)
    feat_list_path = out_dir / "feature_names.txt"
    with open(feat_list_path, "w") as fh:
        fh.write("\n".join(feat_cols))
    log.info(f"   Noms features : {feat_list_path}")

    # Retourne train + test (comme précédemment pour compatibilité)
    return train, test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    horizon = cfg["project"]["horizon_hours"]
    build_features(cfg, horizon_h=horizon)
