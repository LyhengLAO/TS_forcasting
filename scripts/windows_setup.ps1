# ============================================================
# scripts/windows_setup.ps1
# Script d'installation et de configuration pour Windows
#
# Prérequis : PowerShell 7+ (pwsh) ou Windows PowerShell 5.1
# Exécuter en tant qu'administrateur pour les installations globales.
#
# Usage :
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\scripts\windows_setup.ps1
#   .\scripts\windows_setup.ps1 -SkipDocker -SkipKubernetes
# ============================================================

param(
    [switch]$SkipChocolatey,
    [switch]$SkipPython,
    [switch]$SkipDocker,
    [switch]$SkipKubernetes,
    [switch]$SkipMLflow,
    [switch]$OnlyPip,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"

# ─── Couleurs et helpers ──────────────────────────────────────

function Write-Step {
    param($msg)
    Write-Host "`n[STEP] $msg" -ForegroundColor Cyan
}

function Write-OK {
    param($msg)
    Write-Host "  [OK] $msg" -ForegroundColor Green
}

function Write-Warn {
    param($msg)
    Write-Host "  [WARN] $msg" -ForegroundColor Yellow
}

function Write-Err {
    param($msg)
    Write-Host "  [ERR] $msg" -ForegroundColor Red
}

function Write-Info {
    param($msg)
    Write-Host "  [INFO] $msg" -ForegroundColor Gray
}

function Test-Command {
    param($Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-Step {
    param($Name, [scriptblock]$Block)
    Write-Step $Name
    try {
        & $Block
        Write-OK "$Name terminé"
    } catch {
        Write-Err "$Name échoué : $_"
        if (-not $ContinueOnError) { throw }
    }
}

# ─── En-tête ──────────────────────────────────────────────────

Write-Host @"

╔══════════════════════════════════════════════════════════╗
║         ts-forecast-project — Setup Windows              ║
║              Séries Temporelles MLOps                    ║
╚══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Magenta

Write-Info "PowerShell version : $($PSVersionTable.PSVersion)"
Write-Info "OS                 : $([System.Environment]::OSVersion.VersionString)"
Write-Info "Utilisateur        : $env:USERNAME"
Write-Info "Répertoire         : $(Get-Location)"

# ─── Vérification de l'Admin ──────────────────────────────────

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warn "Non-administrateur : certaines installations globales peuvent échouer."
    Write-Info "Pour les outils système, relancer en tant qu'administrateur."
}

# ─── 1. Chocolatey ────────────────────────────────────────────

if (-not $SkipChocolatey -and -not $OnlyPip) {
    Invoke-Step "Installation Chocolatey (package manager Windows)" {
        if (Test-Command "choco") {
            Write-Info "Chocolatey déjà installé : $(choco --version)"
        } else {
            Write-Info "Installation de Chocolatey..."
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
            refreshenv
            Write-OK "Chocolatey installé : $(choco --version)"
        }
    }
}

# ─── 2. Python 3.11 ───────────────────────────────────────────

if (-not $SkipPython -and -not $OnlyPip) {
    Invoke-Step "Python 3.11" {
        if (Test-Command "python") {
            $pyver = python --version 2>&1
            Write-Info "Python détecté : $pyver"
            if ($pyver -notmatch "3\.1[01]") {
                Write-Warn "Version Python recommandée : 3.11.x (actuelle : $pyver)"
            }
        } else {
            Write-Info "Installation Python 3.11 via Chocolatey..."
            choco install python311 -y --no-progress
            refreshenv
        }
    }

    Invoke-Step "Git" {
        if (Test-Command "git") {
            Write-Info "Git détecté : $(git --version)"
        } else {
            choco install git -y --no-progress
            refreshenv
        }
    }

    Invoke-Step "Make (pour le Makefile)" {
        if (Test-Command "make") {
            Write-Info "Make détecté : $(make --version | Select-Object -First 1)"
        } else {
            choco install make -y --no-progress
            refreshenv
        }
    }
}

# ─── 3. Docker Desktop ────────────────────────────────────────

if (-not $SkipDocker -and -not $OnlyPip) {
    Invoke-Step "Docker Desktop" {
        if (Test-Command "docker") {
            Write-Info "Docker détecté : $(docker --version)"
        } else {
            Write-Info "Téléchargement Docker Desktop..."
            $dockerUrl = "https://desktop.docker.com/win/main/amd64/Docker Desktop Installer.exe"
            $installer  = "$env:TEMP\DockerDesktopInstaller.exe"
            Invoke-WebRequest -Uri $dockerUrl -OutFile $installer -UseBasicParsing
            Write-Info "Installation Docker Desktop (peut prendre quelques minutes)..."
            Start-Process -FilePath $installer -Args "install --quiet" -Wait
            Remove-Item $installer
            Write-Warn "Redémarrer Windows pour finaliser Docker Desktop"
        }
    }
}

# ─── 4. Kubernetes (minikube + kubectl) ───────────────────────

if (-not $SkipKubernetes -and -not $OnlyPip) {
    Invoke-Step "minikube (Kubernetes local)" {
        if (Test-Command "minikube") {
            Write-Info "minikube détecté : $(minikube version --short)"
        } else {
            choco install minikube -y --no-progress
            refreshenv
        }
    }

    Invoke-Step "kubectl" {
        if (Test-Command "kubectl") {
            Write-Info "kubectl détecté : $(kubectl version --client --short 2>&1)"
        } else {
            choco install kubernetes-cli -y --no-progress
            refreshenv
        }
    }
}

# ─── 5. Python : venv et dépendances ─────────────────────────

Invoke-Step "Environnement virtuel Python" {
    if (-not (Test-Path ".venv")) {
        Write-Info "Création de l'environnement virtuel .venv..."
        python -m venv .venv
    }
    Write-Info "Activation de .venv..."
    & .\.venv\Scripts\Activate.ps1
    Write-OK "Environnement virtuel activé : $env:VIRTUAL_ENV"
}

Invoke-Step "Mise à jour pip et outils de base" {
    python -m pip install --upgrade pip setuptools wheel
}

Invoke-Step "Installation des dépendances Python" {
    Write-Info "Installation de requirements.txt (peut prendre 5-10 minutes)..."
    pip install -r requirements.txt
    Write-OK "Dépendances installées"
}

# ─── 6. Vérification des imports critiques ────────────────────

Invoke-Step "Vérification des imports Python" {
    $libs = @(
        "pandas", "numpy", "xgboost", "mlflow", "prefect",
        "fastapi", "evidently", "statsmodels", "optuna",
        "great_expectations", "httpx", "yfinance"
    )
    $failed = @()
    foreach ($lib in $libs) {
        $result = python -c "import $lib; print('ok')" 2>&1
        if ($result -eq "ok") {
            if ($Verbose) { Write-Info "  $lib : ✅" }
        } else {
            $failed += $lib
            Write-Warn "  $lib : ❌"
        }
    }
    if ($failed.Count -eq 0) {
        Write-OK "Tous les imports OK"
    } else {
        Write-Warn "Imports échoués : $($failed -join ', ')"
        Write-Info "Réessayer : pip install $($failed -join ' ')"
    }
}

# ─── 7. Création des répertoires ──────────────────────────────

Invoke-Step "Création de la structure de répertoires" {
    $dirs = @(
        "data\raw", "data\interim", "data\processed",
        "models\best_model",
        "reports\figures", "reports\monitoring",
        "logs", "mlruns\artifacts"
    )
    foreach ($dir in $dirs) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    # Fichiers .gitkeep
    @("data\raw", "data\interim", "data\processed", "models") | ForEach-Object {
        New-Item -ItemType File -Force -Path "$_\.gitkeep" | Out-Null
    }
    Write-OK "Répertoires créés"
}

# ─── 8. Copier .env.example → .env.local ──────────────────────

Invoke-Step "Fichier .env.local" {
    if (-not (Test-Path ".env.local")) {
        Copy-Item ".env.example" ".env.local"
        Write-Info "Fichier .env.local créé depuis .env.example"
        Write-Warn "Éditer .env.local avec vos valeurs (MLFLOW_TRACKING_URI, etc.)"
    } else {
        Write-Info ".env.local existe déjà"
    }
}

# ─── 9. MLflow local ──────────────────────────────────────────

if (-not $SkipMLflow) {
    Invoke-Step "Démarrage MLflow (mode SQLite local)" {
        Write-Info "MLflow sera accessible sur http://localhost:5000"
        Write-Info "Pour le démarrer manuellement :"
        Write-Host @"
    # Dans un terminal séparé :
    .\.venv\Scripts\Activate.ps1
    mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
"@ -ForegroundColor DarkGray
    }
}

# ─── 10. Résumé final ─────────────────────────────────────────

Write-Host @"

╔══════════════════════════════════════════════════════════╗
║                  INSTALLATION TERMINÉE                   ║
╚══════════════════════════════════════════════════════════╝

Prochaines étapes :

1. Activer l'environnement virtuel :
   .\.venv\Scripts\Activate.ps1

2. Démarrer MLflow (terminal séparé) :
   mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

3. Lancer le pipeline complet :
   make all
   # ou step by step :
   make collect    # télécharge OpenMeteo + yfinance
   make engineer   # preprocessing + features
   make stats      # tests statistiques
   make train      # entraîne XGBoost (Optuna)
   make test       # lance les tests pytest

4. Démarrer l'API :
   make api       # http://localhost:8000/docs

5. Stack complète Docker :
   make docker-up  # MLflow + API + Grafana + Prometheus

"@ -ForegroundColor Green

Write-OK "Setup terminé ! Bon travail 🚀"
