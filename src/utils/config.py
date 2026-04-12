# ============================================================
# src/utils/config.py
# Chargement, validation et accès à la configuration YAML
# ============================================================

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Erreur de configuration."""


# --- Chargement ---

@lru_cache(maxsize=4)
def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Charge un fichier YAML et le met en cache.
    Les variables d'environnement préfixées TSF_ surchargent
    les valeurs du fichier (utile pour Docker/K8s).

    Ex : TSF_MLFLOW_TRACKING_URI=http://mlflow:5000
         → surcharge cfg["mlflow"]["tracking_uri"]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {p.resolve()}\n"
            "Assurez-vous d'exécuter les commandes depuis la racine du projet."
        )

    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Surcharge via variables d'environnement TSF_*
    cfg = _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: dict) -> dict:
    """
    Surcharge récursive via les variables d'environnement.
    Format : TSF_SECTION_KEY=value
    Ex : TSF_MLFLOW_TRACKING_URI=http://localhost:5001
    """
    for key, val in os.environ.items():
        if not key.startswith("TSF_"):
            continue
        parts = key[4:].lower().split("_", 1)   # TSF_MLFLOW_URI → ["mlflow", "uri"]
        if len(parts) == 2:
            section, subkey = parts
            if section in cfg and isinstance(cfg[section], dict):
                if subkey in cfg[section]:
                    # Conversion de type simple
                    original = cfg[section][subkey]
                    try:
                        if isinstance(original, bool):
                            cfg[section][subkey] = val.lower() in ("true", "1", "yes")
                        elif isinstance(original, int):
                            cfg[section][subkey] = int(val)
                        elif isinstance(original, float):
                            cfg[section][subkey] = float(val)
                        else:
                            cfg[section][subkey] = val
                    except (ValueError, TypeError):
                        cfg[section][subkey] = val
    return cfg


# --- Accès imbriqué ---

def get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """
    Accès sécurisé à une clé imbriquée.

    Exemple :
        get_nested(cfg, "mlflow", "tracking_uri")
        → cfg["mlflow"]["tracking_uri"]
    """
    current = cfg
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current


# --- Validation ---

REQUIRED_KEYS = {
    "project":   ["name", "target", "horizon_hours"],
    "data":      ["start_date", "end_date"],
    "openmeteo": ["latitude", "longitude", "variables"],
    "yfinance":  ["tickers"],
    "paths":     ["raw", "interim", "processed", "models"],
    "mlflow":    ["tracking_uri", "experiment_name"],
}


def validate_config(cfg: dict) -> None:
    """
    Vérifie que tous les champs requis sont présents.
    Lève ConfigError avec un message détaillé sinon.
    """
    errors = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            errors.append(f"Section manquante : [{section}]")
            continue
        for key in keys:
            if key not in cfg[section]:
                errors.append(f"Clé manquante : [{section}].{key}")

    # Vérifications sémantiques
    from datetime import datetime
    try:
        start = datetime.strptime(cfg["data"]["start_date"], "%Y-%m-%d")
        end   = datetime.strptime(cfg["data"]["end_date"],   "%Y-%m-%d")
        if start >= end:
            errors.append("data.start_date doit être antérieure à data.end_date")
        if (end - start).days < 365:
            errors.append("Période minimale : 1 an de données")
    except (KeyError, ValueError):
        pass

    if cfg.get("openmeteo", {}).get("latitude") is not None:
        lat = cfg["openmeteo"]["latitude"]
        lon = cfg["openmeteo"]["longitude"]
        if not (-90 <= lat <= 90):
            errors.append(f"Latitude invalide : {lat} (doit être entre -90 et 90)")
        if not (-180 <= lon <= 180):
            errors.append(f"Longitude invalide : {lon} (doit être entre -180 et 180)")

    if errors:
        msg = "\n".join(f"  • {e}" for e in errors)
        raise ConfigError(f"Configuration invalide :\n{msg}")


# --- Helpers raccourcis ---

def get_paths(cfg: dict) -> dict[str, Path]:
    """Retourne tous les chemins configurés comme objets Path."""
    return {k: Path(v) for k, v in cfg.get("paths", {}).items()}


def get_target(cfg: dict) -> str:
    return cfg["project"]["target"]


def get_horizon(cfg: dict) -> int:
    return cfg["project"]["horizon_hours"]


def get_target_col(cfg: dict) -> str:
    return f"target_{get_horizon(cfg)}h"


# --- Point d'entrée (validation manuelle) ---

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    try:
        cfg = load_config(path)
        validate_config(cfg)
        print(f"✅ Config valide : {path}")
        print(f"   Projet  : {cfg['project']['name']}")
        print(f"   Cible   : {cfg['project']['target']}")
        print(f"   Horizon : {cfg['project']['horizon_hours']}h")
        print(f"   Période : {cfg['data']['start_date']} → {cfg['data']['end_date']}")
    except (FileNotFoundError, ConfigError) as e:
        print(f"❌ {e}")
        sys.exit(1)
