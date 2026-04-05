# ============================================================
# scripts/init_project.py
# Initialisation complète du projet : Git, DVC, structure
#
# Usage :
#   python scripts/init_project.py
#   python scripts/init_project.py --remote-url https://github.com/user/repo.git
#   python scripts/init_project.py --dvc-gdrive <GDRIVE_FOLDER_ID>
# ============================================================

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run(cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Exécute une commande shell."""
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True, cwd=ROOT,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"  ❌ Erreur ({result.returncode})")
        if result.stderr:
            print(f"  {result.stderr}")
        sys.exit(1)
    return result


def step(msg: str) -> None:
    print(f"\n{'─'*50}")
    print(f"▶ {msg}")
    print(f"{'─'*50}")


def check(msg: str) -> None:
    print(f"  ✅ {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


# ─── Répertoires du projet ────────────────────────────────────

DIRECTORIES = [
    "data/raw", "data/interim", "data/processed",
    "models/best_model",
    "reports/figures", "reports/monitoring",
    "logs", "mlruns/artifacts",
    "notebooks",
]

GITKEEP_DIRS = [
    "data/raw", "data/interim", "data/processed",
    "models", "logs",
    "reports/figures", "reports/monitoring",
]


def create_directories() -> None:
    step("Création des répertoires")
    for d in DIRECTORIES:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for d in GITKEEP_DIRS:
        gitkeep = ROOT / d / ".gitkeep"
        gitkeep.touch(exist_ok=True)
    check(f"{len(DIRECTORIES)} répertoires créés")


# ─── Git ──────────────────────────────────────────────────────

def init_git(remote_url: str = None) -> None:
    step("Initialisation Git")

    git_dir = ROOT / ".git"
    if git_dir.exists():
        warn("Dépôt Git déjà initialisé")
    else:
        run("git init")
        check("Git initialisé")

    # Configuration de base
    run('git config core.autocrlf input')    # LF en repo (important sur Windows)
    run('git config core.eol lf')

    if remote_url:
        # Vérifier si remote existe
        result = run("git remote -v", capture=True, check=False)
        if "origin" not in result.stdout:
            run(f"git remote add origin {remote_url}")
            check(f"Remote ajouté : {remote_url}")
        else:
            warn("Remote 'origin' existe déjà")

    # Premier commit
    result = run("git log --oneline -1", capture=True, check=False)
    if result.returncode != 0:
        run("git add configs/ params.yaml dvc.yaml .gitignore README.md "
            "requirements.txt Makefile pyproject.toml pytest.ini "
            ".env.example .dvcignore")
        run('git commit -m "chore: initial project structure"')
        check("Premier commit créé")
    else:
        warn("Commits déjà existants — skip initial commit")


# ─── DVC ──────────────────────────────────────────────────────

def init_dvc(gdrive_folder_id: str = None) -> None:
    step("Initialisation DVC")

    dvc_dir = ROOT / ".dvc"
    if dvc_dir.exists():
        warn("DVC déjà initialisé")
    else:
        run("dvc init")
        check("DVC initialisé")

        # Commit les fichiers DVC
        run("git add .dvc/ .dvcignore")
        run('git commit -m "chore: initialize DVC"')

    # Configuration du remote DVC
    if gdrive_folder_id:
        step(f"Configuration remote DVC (Google Drive : {gdrive_folder_id})")
        run(f"dvc remote add -d gdrive gdrive://{gdrive_folder_id}")
        run("dvc remote modify gdrive gdrive_acknowledge_abuse true")
        run("git add .dvc/config")
        run('git commit -m "feat: configure DVC remote (Google Drive)"')
        check(f"Remote DVC Google Drive configuré : {gdrive_folder_id}")
        print("\n  ℹ️  Première utilisation : dvc push (authentification Google)")
    else:
        warn("Aucun remote DVC configuré — les données ne seront pas partagées")
        print("  Pour configurer un remote Google Drive (gratuit, sans clé API) :")
        print("    python scripts/init_project.py --dvc-gdrive <GDRIVE_FOLDER_ID>")


# ─── Environnement virtuel ────────────────────────────────────

def setup_venv() -> None:
    step("Environnement virtuel Python")

    venv_dir = ROOT / ".venv"
    if venv_dir.exists():
        warn("Environnement virtuel .venv déjà existant")
    else:
        run(f"{sys.executable} -m venv .venv")
        check("Environnement virtuel .venv créé")

    # Chemin Python du venv
    if sys.platform == "win32":
        pip = ROOT / ".venv" / "Scripts" / "pip.exe"
    else:
        pip = ROOT / ".venv" / "bin" / "pip"

    run(f"{pip} install --upgrade pip setuptools wheel")
    run(f"{pip} install -r requirements.txt")
    check("Dépendances installées dans .venv")


# ─── Fichier .env.local ───────────────────────────────────────

def setup_env_file() -> None:
    step("Fichier .env.local")
    env_local = ROOT / ".env.local"
    env_example = ROOT / ".env.example"
    if env_local.exists():
        warn(".env.local existe déjà")
    else:
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_local)
            check(".env.local créé depuis .env.example")
            print("  ℹ️  Éditer .env.local avec vos valeurs")
        else:
            warn(".env.example introuvable")


# ─── Vérification finale ──────────────────────────────────────

def verify_setup() -> None:
    step("Vérification de l'installation")

    checks = {
        "Python 3.11+":    lambda: sys.version_info >= (3, 10),
        "Git":             lambda: run("git --version", capture=True, check=False).returncode == 0,
        "DVC":             lambda: run("dvc --version", capture=True, check=False).returncode == 0,
        "configs/":        lambda: (ROOT / "configs" / "config.yaml").exists(),
        "dvc.yaml":        lambda: (ROOT / "dvc.yaml").exists(),
        "params.yaml":     lambda: (ROOT / "params.yaml").exists(),
        "data/raw/":       lambda: (ROOT / "data" / "raw").exists(),
        "data/processed/": lambda: (ROOT / "data" / "processed").exists(),
    }

    all_ok = True
    for name, fn in checks.items():
        try:
            ok = fn()
        except Exception:
            ok = False
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  🎉 Tout est en ordre !")
    else:
        print("\n  ⚠️  Certains checks ont échoué — voir ci-dessus")


# ─── Affichage des prochaines étapes ─────────────────────────

def print_next_steps() -> None:
    print("""
╔══════════════════════════════════════════════════════════╗
║              PROJET INITIALISÉ AVEC SUCCÈS               ║
╚══════════════════════════════════════════════════════════╝

Prochaines étapes :

  Windows (PowerShell) :
    .\.venv\Scripts\Activate.ps1

  Linux / macOS :
    source .venv/bin/activate

  Puis :
    make collect    → collecter OpenMeteo + yfinance (sans clé API)
    make engineer   → preprocessing + feature engineering
    make stats      → tests ADF, KPSS, STL, Granger
    make train      → entraîner + tracker avec MLflow
    make test       → tests pytest

  Interface MLflow :
    mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
    → http://localhost:5000

  Documentation API :
    make api
    → http://localhost:8000/docs
""")


# ─── Point d'entrée ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Initialisation du projet ts-forecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--remote-url",    default=None,
                        help="URL du dépôt Git distant (ex: https://github.com/user/repo.git)")
    parser.add_argument("--dvc-gdrive",    default=None,
                        help="ID du dossier Google Drive pour le remote DVC")
    parser.add_argument("--no-venv",       action="store_true",
                        help="Ne pas créer l'environnement virtuel (si déjà configuré)")
    parser.add_argument("--no-dvc",        action="store_true",
                        help="Ne pas initialiser DVC")
    args = parser.parse_args()

    print("\n🚀 Initialisation du projet ts-forecast-project\n")

    create_directories()
    init_git(remote_url=args.remote_url)

    if not args.no_dvc:
        init_dvc(gdrive_folder_id=args.dvc_gdrive)

    if not args.no_venv:
        setup_venv()

    setup_env_file()
    verify_setup()
    print_next_steps()


if __name__ == "__main__":
    main()
