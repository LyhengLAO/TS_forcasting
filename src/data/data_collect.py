# ============================================================
# src/data/data_collect.py
# Collecte de données multi-sources:
#   - OpenMeteo  : météo horaire (archive + forecast)
#   - yfinance   : séries financières (CAC40, énergie)
# ============================================================

import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest import result

import httpx
import pandas as pd
import numpy as np
import yaml
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# --- Chargement config ---
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
    
# --- OpenMeteo ---
def _build_openmeteo_params(cfg: dict) -> dict:
    """Construit les paramètres de la requête OpenMeteo."""
    om = cfg["openmeteo"]
    return {
        "latitude":   om["latitude"],
        "longitude":  om["longitude"],
        "start_date": cfg["data"]["start_date"],
        "end_date":   cfg["data"]["end_date"],
        "hourly":     om["variables"],
        "timezone":   om.get("timezone", "Europe/Paris"),
        "wind_speed_unit": "ms",         # m/s (plus précis que km/h)
        "precipitation_unit": "mm",
    }

def fetch_openmeteo_data(cfg: dict, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """
    Télécharge les données météo horaires depuis l'API OpenMeteo Archive.
    Gère automatiquement les gros intervalles en découpant par année.
    """
    om = cfg["openmeteo"]
    url   = om.get("base_url", "https://archive-api.open-meteo.com/v1/archive")
    start = datetime.strptime(cfg["data"]["start_date"], "%Y-%m-%d") # date de début de la collecte
    end   = datetime.strptime(cfg["data"]["end_date"],   "%Y-%m-%d") # date de fin de la collecte

    log.info(f"OpenMeteo : {start.date()} -> {end.date()}")
    log.info(f"Variables : {om['variables']}")

    frames = [] 
    current = start # pointeur de date
    while current < end:
        year_end = min(datetime(current.year, 12, 31), end)
        params = _build_openmeteo_params(cfg)
        params["start_date"] = current.strftime("%Y-%m-%d")
        params["end_date"]   = year_end.strftime("%Y-%m-%d")

        for attempt in range(retries):
            try:
                with httpx.Client(
                    timeout=cfg["openmeteo"].get("timeout_seconds", 60)
                ) as client:
                    resp = client.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                hourly = data.get("hourly", {})
                if not hourly:
                    raise ValueError(f"Réponse vide pour {params['start_date']}")

                df_year = pd.DataFrame(hourly)
                df_year["time"] = pd.to_datetime(df_year["time"])
                df_year = df_year.set_index("time").rename_axis("datetime")
                df_year.columns = [f"meteo_{c}" for c in df_year.columns]

                frames.append(df_year)
                log.info(f"  {current.year} : {len(df_year)} observations")
                break   # succès

            except (httpx.HTTPError, httpx.TimeoutException) as e:
                log.warning(f"  Tentative {attempt+1}/{retries} échouée : {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise

        current = datetime(current.year + 1, 1, 1)

    df = pd.concat(frames).sort_index()

    # Vérification de la continuité temporelle
    expected_hours = int((end - start).total_seconds() / 3600) + 1
    actual_hours   = len(df)
    missing_pct    = max(0, (expected_hours - actual_hours) / expected_hours * 100)
    if missing_pct > 5:
        log.warning(f"{missing_pct:.1f}% des heures manquantes dans OpenMeteo")

    # Supprimer les doublons éventuels
    df = df[~df.index.duplicated(keep="first")]

    log.info(
        f"OpenMeteo : {len(df)} obs | "
        f"{df.index.min().date()} → {df.index.max().date()} | "
        f"Nulls max : {df.isnull().mean().max() * 100:.1f}%"
    )
    return df

# --- yfinance ---
def fetch_yfinance_data(cfg: dict) -> pd.DataFrame:
    """
    Télécharge les données financières depuis yfinance.
    Données journalières -> resample horaire par forward-fill.
    """
    tickers  = cfg["yfinance"]["tickers"]
    start    = cfg["data"]["start_date"]
    end      = cfg["data"]["end_date"]
    interval = cfg["yfinance"].get("interval", "1d")

    log.info(f"yfinance : {tickers}")

    frames = []
    for ticker in tickers:
        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    timeout=30,
                )
                if raw.empty:
                    log.warning(f"{ticker} : données vides")
                    break

                # Nettoyage de MultiIndex (si présent)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.droplevel(1)

                # Renommer les colonnes avec préfixe ticker
                prefix = ticker.replace("^", "").replace(".", "_").lower()
                raw.columns = [f"fin_{prefix}_{c.lower()}" for c in raw.columns]
                raw.index.name = "datetime"
                raw.index = pd.to_datetime(raw.index)

                # Resample : journalier → horaire (forward fill, max 24h)
                raw = raw.resample("h").ffill(limit=24)

                frames.append(raw)
                log.info(f"{ticker} : {len(raw)} observations (horaire)")
                break

            except Exception as e:
                log.warning(f"  {ticker} tentative {attempt+1}/3 : {e}")
                if attempt < 2:
                    time.sleep(3)

    if not frames:
        log.error("Aucune donnée yfinance récupérée")
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    df.index.name = "datetime"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    log.info(
        f"yfinance : {len(df)} obs | "
        f"{len(df.columns)} colonnes | "
        f"Nulls max : {df.isnull().mean().max() * 100:.1f}%"
    )
    return df

# --- Vérification de sanité des données ---
def sanity_check(df: pd.DataFrame, name: str) -> None:
    """Vérifications basiques après collecte."""
    log.info(f"\n{'─'*45}")
    log.info(f"Sanity check : {name}")
    log.info(f"  Shape      : {df.shape}")
    log.info(f"  Index      : {df.index.min()} -> {df.index.max()}")
    log.info(f"  Freq       : {pd.infer_freq(df.index[:100]) or 'irrégulière'}")
    log.info(f"  Nulls (%)  : {(df.isnull().mean() * 100).max():.2f}% max")
    log.info(f"  Dtype      : {df.dtypes.value_counts().to_dict()}")

    high_null_cols = df.columns[df.isnull().mean() > 0.10].tolist()
    if high_null_cols:
        log.warning(f"Colonnes avec >10% des valeurs nulles(manquantes) : {high_null_cols}")

# --- Pipeline de collecte complète ---
def collect_all_data(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    """
    Pipeline complet de collecte.
    Retourne un dict avec les DataFrames bruts.
    Sauvegarde en format Parquet (compressé snappy).
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 50)
    log.info("COLLECTE DES DONNÉES")
    log.info(f"  Période      : {cfg['data']['start_date']} -> {cfg['data']['end_date']}")
    log.info(f"  Localisation : ({cfg['openmeteo']['latitude']}, {cfg['openmeteo']['longitude']})")
    log.info("=" * 50)

    results = {}

    # OpenMeteo
    df_meteo = fetch_openmeteo_data(cfg)
    sanity_check(df_meteo, "OpenMeteo")
    weather_path = raw_dir / "weather_raw.parquet"
    df_meteo.to_parquet(weather_path, compression="snappy")
    log.info(f"Sauvegardé : {weather_path} ({weather_path.stat().st_size / 1e6:.1f} MB)")
    results["meteo"] = df_meteo

    # yfinance
    df_finance = fetch_yfinance_data(cfg)
    if not df_finance.empty:
        sanity_check(df_finance, "yfinance")
        finance_path = raw_dir / "finance_raw.parquet"
        df_finance.to_parquet(finance_path, compression="snappy")
        log.info(f"Sauvegardé : {finance_path} ({finance_path.stat().st_size / 1e6:.1f} MB)")
        results["finance"] = df_finance
    else:
        log.warning("yfinance vide — continuation sans données financières")
        results["finance"] = pd.DataFrame()

    log.info("Collecte terminée.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collecte de données météo et financières")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Chemin vers le fichier de configuration YAML",
    )
    args = parser.parse_args()
    collect_all_data(args.config)