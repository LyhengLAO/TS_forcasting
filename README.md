# ts-forecast-project

Projet MLOps complet de prévision de séries temporelles météo/finance.  
**Sans aucune clé API** — données OpenMeteo (météo horaire) + yfinance (indices financiers).  
**Compatible Windows** avec PowerShell / make.

---

## Architecture du projet

```
ts-forecast-project/
├── configs/               ← Configuration YAML centralisée
│   ├── config.yaml        ← Config principale (données, chemins, MLflow)
│   ├── model_config.yaml  ← Hyperparamètres et seuils de production
│   └── monitoring_config.yaml ← Drift, alertes, Prometheus
├── data/                  ← Versionnées par DVC (non committées Git)
│   ├── raw/               ← Données brutes (OpenMeteo, yfinance)
│   ├── interim/           ← Données nettoyées
│   └── processed/         ← Features finales
├── notebooks/             ← Exploration → scripts finaux
├── src/
│   ├── data/              ← Collecte, validation, preprocessing
│   ├── features/          ← Feature engineering time series
│   ├── models/            ← Train, evaluate, select
│   ├── analysis/          ← Tests statistiques (ADF, KPSS, STL, Granger)
│   └── monitoring/        ← Drift detection (Evidently)
├── pipelines/             ← Pipelines Prefect
├── api/                   ← API FastAPI de prédiction
├── mlops/
│   ├── docker/            ← Dockerfile + docker-compose
│   └── kubernetes/        ← Manifests K8s + HPA
├── tests/                 ← Tests pytest
├── .github/workflows/     ← CI/CD GitHub Actions
├── dvc.yaml               ← DAG DVC (reproductibilité)
└── params.yaml            ← Hyperparamètres versionnés DVC
```

---

## Démarrage rapide (Windows)

### 1. Prérequis

```powershell
# Chocolatey (package manager Windows)
choco install python311 git make docker-desktop minikube kubernetes-cli

# Ou via winget
winget install Python.Python.3.11 Git.Git
```

### 2. Installation

```bash
git clone https://github.com/LyhengLAO/TS_forcasting
cd ts-forecast-project

make setup        # installe toutes les dépendances pip
make dirs         # crée les répertoires data/, models/, logs/
```

### 3. Pipeline complet en une commande

```bash
make all
# Équivalent à : make collect → engineer → stats → train → test
```

### 4. Étape par étape

```bash
# Phase 1 : Collecte (OpenMeteo + yfinance, aucune clé API)
make collect

# Phase 2 : Data engineering
make engineer

# Phase 3 : Tests statistiques (ADF, KPSS, STL, Granger)
make stats

# Phase 4 : Entraînement (Optuna + MLflow tracking)
make mlflow     # dans un autre terminal : démarrer MLflow
make train

# Phase 5 : Tests
make test

# Phase 5 : MLOps — Docker + Kubernetes
make docker-build
make docker-up
make k8s-start
make k8s-deploy
```

---

## Sources de données (sans clé API)

| Source | Type | Variables | Fréquence |
|--------|------|-----------|-----------|
| [OpenMeteo Archive](https://open-meteo.com/en/docs/historical-weather-api) | Météo | Température, humidité, vent, précipitations, radiation | Horaire |
| [yfinance](https://pypi.org/project/yfinance/) | Finance | CAC40, EDF, TotalEnergies, S&P500 | Journalier → horaire |

---

## Modèles comparés

| Modèle | Type | Usage |
|--------|------|-------|
| **XGBoost** | Gradient Boosting | Modèle principal (tabular time series) |
| **SARIMA** | Statistique | Baseline univarié (p,d,q)(P,D,Q,s) |
| **Prophet** | Statistique ML | Saisonnalités multiples, régresseurs |
| **LSTM** | Deep Learning | Apprentissage de dépendances longues |

Sélection automatique via **Optuna** (50 trials, TPE sampler).

---

## Stack MLOps

| Composant | Outil |
|-----------|-------|
| Versioning données | **DVC** |
| Experiment tracking | **MLflow** |
| Orchestration | **Prefect** |
| Hyperparameter tuning | **Optuna** |
| Data drift | **Evidently** |
| API de serving | **FastAPI** |
| Conteneurisation | **Docker** |
| Orchestration K8s | **Kubernetes + minikube** |
| Autoscaling | **HPA (CPU/mémoire)** |
| Métriques | **Prometheus + Grafana** |
| CI/CD | **GitHub Actions** |

---

## Tests statistiques implémentés

- **ADF + KPSS** : détection stationnarité + type de non-stationnarité
- **Décomposition STL** : tendance, saisonnalité (24h et 168h), résidus
- **ACF / PACF** : suggestion des ordres p et q pour ARIMA
- **Causalité de Granger** : quelles features prédivent la cible ?
- **Ljung-Box** : autocorrélation des résidus
- **Shapiro-Wilk** : normalité des résidus
- **Durbin-Watson** : autocorrélation d'ordre 1

---

## Monitoring en production

- Drift détecté via **Evidently** (PSI, KS test, Jensen-Shannon)
- Alerte si > 20% des features ont drifté
- Métriques exposées via `/metrics` (Prometheus)
- Dashboard Grafana sur `http://localhost:3000`
- Re-entraînement déclenché automatiquement (configurable)

---

## Configuration DVC remote (optionnel)

Pour partager les données d'équipe sans S3 (gratuit) :

```bash
# Google Drive (sans clé API)
dvc remote add -d gdrive gdrive://<VOTRE_FOLDER_ID>
dvc push   # upload données
dvc pull   # télécharger sur un autre poste
```

---

## Variables d'environnement

Créer un fichier `.env.local` (jamais commité) :

```env
MLFLOW_TRACKING_URI=http://localhost:5000
PREFECT_API_URL=http://localhost:4200/api
LOG_LEVEL=INFO
```

---

## Contribuer

```bash
git checkout -b feature/ma-feature
make format    # formater le code
make lint      # vérifier qualité
make test      # tester
git push origin feature/ma-feature
# Ouvrir une Pull Request → CI se déclenche automatiquement
```
