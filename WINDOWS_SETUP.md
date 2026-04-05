# Guide d'installation Windows — ts-forecast-project

Guide complet pour configurer le projet sur Windows 10/11,
**sans aucune clé API**, données OpenMeteo + yfinance uniquement.

---

## Prérequis système

- Windows 10 (version 2004+) ou Windows 11
- 8 Go RAM minimum (16 Go recommandé pour le LSTM)
- 20 Go d'espace disque libre
- Connexion internet

---

## Option A — Installation automatique (recommandée)

```powershell
# 1. Ouvrir PowerShell en tant qu'administrateur
# Clic droit sur PowerShell → "Exécuter en tant qu'administrateur"

# 2. Autoriser les scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Cloner le projet
git clone https://github.com/votre-org/ts-forecast-project.git
cd ts-forecast-project

# 4. Lancer le script d'installation
.\scripts\windows_setup.ps1

# 5. Initialiser Git + DVC
python scripts/init_project.py
```

---

## Option B — Installation manuelle étape par étape

### 1. Chocolatey (package manager Windows)

```powershell
# En tant qu'administrateur :
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### 2. Outils système

```powershell
# Python 3.11, Git, Make
choco install python311 git make -y

# Recharger le PATH
refreshenv

# Vérification
python --version   # Python 3.11.x
git --version
make --version
```

### 3. Docker Desktop

```powershell
choco install docker-desktop -y
# Redémarrer Windows après l'installation
# Activer WSL2 si demandé (recommandé)
```

Ou télécharger manuellement : https://www.docker.com/products/docker-desktop/

### 4. Kubernetes local (minikube)

```powershell
choco install minikube kubernetes-cli -y
refreshenv

# Démarrer minikube avec le driver Docker
minikube start --driver=docker --cpus=2 --memory=4096
```

### 5. Environnement Python

```powershell
# Dans le répertoire du projet
cd ts-forecast-project

# Créer et activer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Vérifier l'activation (le prompt change)
# (.venv) PS C:\...\ts-forecast-project>

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Vérifier les imports clés
python -c "import pandas, xgboost, mlflow, fastapi, evidently; print('OK')"
```

---

## Problèmes courants Windows

### ❌ `torch` (PyTorch) ne s'installe pas

PyTorch peut nécessiter une version spécifique selon votre GPU :

```powershell
# CPU uniquement (tous les PC)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU NVIDIA (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### ❌ `prophet` échoue à l'installation

Prophet dépend de `pystan` ou `cmdstanpy` :

```powershell
pip install prophet --no-build-isolation
# Si ça échoue :
pip install pystan==2.19.1.1
pip install prophet
```

### ❌ Erreur `make` non reconnu

```powershell
# Via Chocolatey
choco install make -y
refreshenv

# Ou utiliser les commandes Python directement :
python -m src.data.collect       # au lieu de make collect
python -m src.features.build_features  # au lieu de make engineer
```

### ❌ Caractères spéciaux dans les logs (emojis)

```powershell
# Activer UTF-8 dans PowerShell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

### ❌ `docker-compose` non reconnu

Depuis Docker Desktop v4+, la commande est `docker compose` (sans tiret) :

```powershell
# Ancien format
docker-compose up -d

# Nouveau format (Docker Desktop 4+)
docker compose up -d
```

### ❌ Ports 5000 ou 8000 déjà utilisés

```powershell
# Trouver quel processus utilise le port 5000
netstat -ano | findstr :5000
# Ou :
Get-NetTCPConnection -LocalPort 5000 | Select-Object -ExpandProperty OwningProcess | Get-Process

# Tuer le processus si nécessaire
taskkill /PID <PID> /F
```

### ❌ Permissions sur le répertoire `.venv`

```powershell
# Réinitialiser les permissions
icacls .venv /reset /T
# Puis recréer :
Remove-Item -Recurse -Force .venv
python -m venv .venv
```

---

## Démarrage du projet (pipeline complet)

```powershell
# Activer l'environnement (à faire à chaque nouvelle session)
.\.venv\Scripts\Activate.ps1

# ── Terminal 1 : MLflow ──
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
# → http://localhost:5000

# ── Terminal 2 : Pipeline ──
make collect    # Collecte OpenMeteo + yfinance (~2-5 min)
make engineer   # Preprocessing + features (~1-3 min)
make stats      # Tests statistiques (~1 min)
make train      # Entraînement Optuna 50 trials (~10-30 min)
make test       # Tests pytest

# ── Terminal 3 : API ──
make api
# → http://localhost:8000/docs

# ── Ou tout en une commande ──
make all
```

---

## Stack Docker Compose (dev local)

```powershell
# Builder les images
make docker-build

# Démarrer MLflow + API + Prometheus + Grafana
make docker-up

# Services disponibles :
# MLflow  : http://localhost:5000
# API     : http://localhost:8000/docs
# Metrics : http://localhost:8000/metrics
# Prometheus : http://localhost:9090
# Grafana : http://localhost:3000  (admin / admin_dev)

# Logs de l'API
docker compose -f mlops/docker/docker-compose.yml logs -f api

# Arrêter tout
make docker-down
```

---

## Kubernetes local (minikube)

```powershell
# Démarrer minikube
minikube start --driver=docker --cpus=2 --memory=4096
minikube addons enable ingress

# Builder l'image dans minikube (évite de pousser sur un registry)
minikube image build -f mlops/docker/Dockerfile.api -t ts-forecast-api:latest .

# Déployer
kubectl apply -f mlops/kubernetes/namespace.yaml
kubectl apply -f mlops/kubernetes/configmap.yaml
kubectl apply -f mlops/kubernetes/service.yaml
kubectl apply -f mlops/kubernetes/deployment.yaml

# Vérifier
kubectl get pods -n ts-forecast
kubectl get svc -n ts-forecast

# Accéder à l'API
minikube service ts-forecast-service -n ts-forecast --url
# → http://127.0.0.1:<PORT>

# Dashboard Kubernetes
minikube dashboard

# Arrêter
minikube stop
```

---

## DVC — Versioning des données

```powershell
# Initialiser DVC (fait automatiquement par scripts/init_project.py)
dvc init

# Remote Google Drive (gratuit, sans clé API)
dvc remote add -d gdrive gdrive://<VOTRE_FOLDER_ID>
# VOTRE_FOLDER_ID = ID dans l'URL Google Drive

# Versionner les données
dvc add data/raw/weather_raw.parquet
dvc add data/raw/finance_raw.parquet
git add data/raw/*.dvc .gitignore
git commit -m "data: add raw data DVC tracking"
dvc push   # envoyer vers Google Drive

# Sur un autre PC :
git pull
dvc pull   # récupérer les données depuis Google Drive
```

---

## Commandes make sur Windows

Si `make` n'est pas disponible, voici les équivalents PowerShell :

| `make ...`   | Équivalent PowerShell |
|---|---|
| `make collect` | `python -m src.data.collect` |
| `make engineer` | `python -m src.data.preprocess; python -m src.features.build_features` |
| `make stats` | `python -m src.analysis.statistical_tests` |
| `make train` | `python -m src.models.train` |
| `make test` | `pytest tests/ -v` |
| `make api` | `uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload` |
| `make monitor` | `python -m src.monitoring.drift` |

---

## Ressources

- [OpenMeteo API](https://open-meteo.com/en/docs/historical-weather-api) — météo historique gratuite
- [yfinance](https://github.com/ranaroussi/yfinance) — données financières Yahoo
- [MLflow docs](https://mlflow.org/docs/latest) — tracking et registry
- [DVC docs](https://dvc.org/doc) — versioning données et modèles
- [Prefect docs](https://docs.prefect.io) — orchestration pipelines
- [Evidently docs](https://docs.evidentlyai.com) — monitoring drift
- [minikube docs](https://minikube.sigs.k8s.io/docs) — Kubernetes local
