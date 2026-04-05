# ============================================================
# src/utils/logger.py
# Logger centralisé : format uniforme, rotation, niveau configurable
# ============================================================

import logging
import logging.handlers
import sys
from pathlib import Path


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Crée ou récupère un logger avec format cohérent.

    Args:
        name        : Nom du logger (ex: "src.data.collect")
        level       : Niveau de log ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file    : Chemin vers le fichier de log (optionnel)
        max_bytes   : Taille max avant rotation (défaut : 10 MB)
        backup_count: Nombre de fichiers de backup gardés

    Returns:
        logging.Logger configuré
    """
    logger = logging.getLogger(name)

    # Éviter la duplication des handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # Handler fichier avec rotation (optionnel)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(logging.DEBUG)   # tout dans le fichier
        logger.addHandler(file_handler)

    # Éviter la propagation vers le root logger
    logger.propagate = False
    return logger


def setup_logging(config: dict = None, log_file: str = "logs/project.log") -> None:
    """
    Configure le logging global depuis la config YAML.
    À appeler une fois au démarrage (pipeline, API, etc.)
    """
    level    = "INFO"
    log_path = log_file

    if config and "logging" in config:
        level    = config["logging"].get("level", "INFO")
        log_path = config["logging"].get("file", log_file)

    # Logger racine du projet
    root = get_logger("ts_forecast", level=level, log_file=log_path)

    # Réduire la verbosité des bibliothèques tierces
    for noisy_lib in ["httpx", "urllib3", "yfinance", "matplotlib", "numba",
                      "prophet", "cmdstanpy", "pystan"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    root.info(f"Logging configuré — niveau={level} | fichier={log_path}")
