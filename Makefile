# ============================================================
# Makefile — ts-forecast-project
# Compatible Windows via : choco install make
# (ou utiliser PowerShell directement pour chaque commande)
#
# Usage :
#   make setup        → Installer les dépendances
#   make collect      → Collecter les données (OpenMeteo + yfinance)
#   make engineer     → Preprocessing + feature engineering
#   make stats        → Tests statistiques
#   make train        → Entraîner le meilleur modèle (Optuna)
#   make test         → Lancer tous les tests pytest
#   make lint         → Vérifier la qualité du code
#   make mlflow       → Démarrer le serveur MLflow
#   make prefect      → Démarrer le serveur Prefect
#   make docker-build → Construire les images Docker
#   make docker-up    → Démarrer la stack complète
#   make k8s-deploy   → Déployer sur Kubernetes (minikube)
#   make monitor      → Rapport de monitoring + drift
#   make dvc-repro    → Rejouer le pipeline DVC complet
#   make clean        → Nettoyer les fichiers temporaires
#   make all          → Tout lancer de A à Z
# ============================================================

PYTHON     := python
PIP        := pip
PYTEST     := pytest
DVC        := dvc
MLFLOW     := mlflow
PREFECT    := prefect
DOCKER     := docker
KUBECTL    := kubectl
MINIKUBE   := minikube

CONFIG     := configs/config.yaml
MODEL_CFG  := configs/model_config.yaml

.PHONY: all setup collect engineer stats train evaluate test lint \
        format mlflow prefect docker-build docker-up docker-down \
        k8s-start k8s-deploy k8s-status k8s-logs monitor \
        dvc-init dvc-repro dvc-push dvc-status \
        clean clean-data clean-models help

# ─── Par défaut ────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ts-forecast-project — Commandes disponibles"
	@echo "  =============================================="
	@echo "  make setup        Installer toutes les dépendances Python"
	@echo "  make collect      Collecter données OpenMeteo + yfinance"
	@echo "  make engineer     Preprocessing + feature engineering"
	@echo "  make stats        Tests statistiques (ADF, KPSS, STL, Granger)"
	@echo "  make train        Entraîner le modèle (Optuna + MLflow)"
	@echo "  make evaluate     Évaluer le modèle sur le jeu de test"
	@echo "  make test         Tests pytest (unit + intégration)"
	@echo "  make lint         Vérifier qualité code (flake8 + black + isort)"
	@echo "  make format       Formater le code (black + isort)"
	@echo "  make mlflow       Démarrer MLflow tracking server"
	@echo "  make prefect      Démarrer Prefect server"
	@echo "  make docker-build Construire images Docker"
	@echo "  make docker-up    Démarrer la stack complète (docker-compose)"
	@echo "  make docker-down  Arrêter la stack"
	@echo "  make k8s-start    Démarrer minikube (Kubernetes local)"
	@echo "  make k8s-deploy   Déployer sur Kubernetes"
	@echo "  make monitor      Rapport drift Evidently"
	@echo "  make dvc-repro    Rejouer le pipeline DVC"
	@echo "  make clean        Nettoyer fichiers temporaires"
	@echo "  make all          Pipeline complet (collect → train → test)"
	@echo ""

# ─── Installation ─────────────────────────────────────────────
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dépendances installées"
	@$(PYTHON) -c "import xgboost, mlflow, prefect, evidently; print('Imports OK')"

setup-dev: setup
	$(PIP) install black isort flake8 mypy pytest-cov
	@echo "Outils dev installés"

# ─── Répertoires ──────────────────────────────────────────────
dirs:
	@mkdir -p data/raw data/interim data/processed
	@mkdir -p models reports/figures reports/monitoring logs
	@echo "" > data/raw/.gitkeep
	@echo "" > data/interim/.gitkeep
	@echo "" > data/processed/.gitkeep
	@echo "" > models/.gitkeep
	@echo "Répertoires créés"

# ─── Données ──────────────────────────────────────────────────
collect: dirs
	@echo "Collecte OpenMeteo + yfinance..."
	$(PYTHON) -m src.data.collect
	@echo "Données collectées dans data/raw/"

validate:
	@echo "Validation des données..."
	$(PYTHON) -m src.data.validate
	@echo "Validation terminée"

preprocess:
	@echo "Preprocessing..."
	$(PYTHON) -m src.data.preprocess
	@echo "Données nettoyées dans data/interim/"

features:
	@echo "Feature engineering..."
	$(PYTHON) -m src.features.build_features
	@echo "Features dans data/processed/"

engineer: validate preprocess features
	@echo "Pipeline data engineering complet"

# ─── Analyse statistique ──────────────────────────────────────
stats:
	@echo "Tests statistiques..."
	$(PYTHON) -m src.analysis.statistical_tests
	@echo "Résultats dans reports/statistical_tests.json"

# ─── Entraînement ─────────────────────────────────────────────
train:
	@echo "Entraînement (Optuna + MLflow)..."
	$(PYTHON) -m src.models.train
	@echo "Modèle sauvegardé dans models/"

evaluate:
	@echo "Évaluation..."
	$(PYTHON) -m src.models.evaluate
	@echo "Métriques dans reports/metrics_test.json"

pipeline-train:
	@echo "Pipeline Prefect d'entraînement..."
	$(PYTHON) -m pipelines.training_pipeline

pipeline-data:
	@echo "Pipeline Prefect de données..."
	$(PYTHON) -m pipelines.data_pipeline

pipeline-infer:
	@echo "Pipeline Prefect d'inférence..."
	$(PYTHON) -m pipelines.inference_pipeline

# ─── Tests ────────────────────────────────────────────────────
test:
	@echo "Tests pytest..."
	$(PYTEST) tests/ -v --tb=short \
		--cov=src --cov=api \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage_html
	@echo "Rapport coverage : reports/coverage_html/index.html"

test-unit:
	$(PYTEST) tests/test_data.py tests/test_models.py -v --tb=short

test-api:
	$(PYTEST) tests/test_api.py -v --tb=short

test-fast:
	$(PYTEST) tests/ -v --tb=short -x   # stoppe au premier échec

# ─── Qualité code ─────────────────────────────────────────────
lint:
	@echo "Lint..."
	flake8 src/ api/ pipelines/ tests/ --max-line-length=120 --extend-ignore=E203,W503
	black --check src/ api/ pipelines/ tests/
	isort --check-only src/ api/ pipelines/ tests/
	@echo "Lint OK"

format:
	@echo "Formatage..."
	black src/ api/ pipelines/ tests/
	isort src/ api/ pipelines/ tests/
	@echo "Code formaté"

# ─── MLflow ───────────────────────────────────────────────────
mlflow:
	@echo "Démarrage MLflow server sur http://localhost:5000"
	$(MLFLOW) server \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root ./mlruns/artifacts \
		--host 127.0.0.1 \
		--port 5000

mlflow-bg:
	@echo "MLflow en arrière-plan..."
	start /B $(MLFLOW) server --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# ─── Prefect ──────────────────────────────────────────────────
prefect:
	@echo "Démarrage Prefect server..."
	$(PREFECT) server start

# ─── API locale ───────────────────────────────────────────────
api:
	@echo "Démarrage API FastAPI sur http://localhost:8000"
	uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

api-docs:
	@echo "Documentation API : http://localhost:8000/docs"

# ─── Docker ───────────────────────────────────────────────────
docker-build:
	@echo "Build images Docker..."
	$(DOCKER) build -f mlops/docker/Dockerfile.api     -t ts-forecast-api:latest     .
	$(DOCKER) build -f mlops/docker/Dockerfile.trainer -t ts-forecast-trainer:latest .
	@echo "✅ Images construites"

docker-up:
	@echo "Démarrage stack Docker Compose..."
	$(DOCKER) compose -f mlops/docker/docker-compose.yml up -d mlflow postgres api prometheus grafana
	@echo "   Stack démarrée"
	@echo "   MLflow  : http://localhost:5000"
	@echo "   API     : http://localhost:8000"
	@echo "   Grafana : http://localhost:3000  (admin/admin_dev)"

docker-train:
	$(DOCKER) compose -f mlops/docker/docker-compose.yml \
		--profile training run --rm trainer

docker-down:
	$(DOCKER) compose -f mlops/docker/docker-compose.yml down
	@echo "Stack arrêtée"

docker-logs:
	$(DOCKER) compose -f mlops/docker/docker-compose.yml logs -f api

# ─── Kubernetes (minikube local) ──────────────────────────────
k8s-start:
	@echo "Démarrage minikube..."
	$(MINIKUBE) start --cpus=2 --memory=4096 --driver=docker
	$(MINIKUBE) addons enable ingress
	@echo "Minikube démarré"

k8s-build-push:
	@echo "Build image dans minikube..."
	$(MINIKUBE) image build -f mlops/docker/Dockerfile.api -t ts-forecast-api:latest .

k8s-deploy:
	@echo "Déploiement Kubernetes..."
	$(KUBECTL) apply -f mlops/kubernetes/configmap.yaml
	$(KUBECTL) apply -f mlops/kubernetes/service.yaml
	$(KUBECTL) apply -f mlops/kubernetes/deployment.yaml
	$(KUBECTL) rollout status deployment/ts-forecast-api --timeout=120s
	@echo "Déploiement OK"

k8s-status:
	$(KUBECTL) get pods,svc,hpa -l app=ts-forecast

k8s-logs:
	$(KUBECTL) logs -l app=ts-forecast -f --tail=100

k8s-delete:
	$(KUBECTL) delete -f mlops/kubernetes/
	@echo "Ressources K8s supprimées"

# ─── DVC ──────────────────────────────────────────────────────
dvc-init:
	git init
	$(DVC) init
	@echo "DVC initialisé"

dvc-repro:
	@echo "DVC repro..."
	$(DVC) repro
	@echo "Pipeline DVC rejoué"

dvc-status:
	$(DVC) status

dvc-dag:
	$(DVC) dag

dvc-params-diff:
	$(DVC) params diff

dvc-metrics:
	$(DVC) metrics show

# DVC remote : Google Drive (gratuit, sans clé API externe)
dvc-remote-gdrive:
	@echo "Configurez le remote Google Drive :"
	@echo "  dvc remote add -d gdrive gdrive://<FOLDER_ID>"
	@echo "  dvc remote modify gdrive gdrive_acknowledge_abuse true"

dvc-push:
	$(DVC) push

dvc-pull:
	$(DVC) pull

# ─── Monitoring ───────────────────────────────────────────────
monitor:
	@echo "Rapport de monitoring..."
	$(PYTHON) -m src.monitoring.drift
	@echo "Rapport dans reports/monitoring/"

# ─── Pipeline complet ─────────────────────────────────────────
all: dirs collect engineer stats train test
	@echo ""
	@echo "   Pipeline complet terminé !"
	@echo "   Modèle    : models/best_model/"
	@echo "   Métriques : reports/metrics_test.json"
	@echo "   Figures   : reports/figures/"
	@echo ""

# ─── Nettoyage ────────────────────────────────────────────────
clean:
	@echo "Nettoyage..."
	-find . -type d -name __pycache__ -exec rm -rf {} + 2>nul || true
	-find . -name "*.pyc" -delete 2>nul || true
	-find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>nul || true
	-rm -rf .pytest_cache/ .coverage htmlcov/ 2>nul || true
	@echo "Nettoyage terminé"

clean-data:
	@echo "Suppression des données (data/)..."
	-rm -f data/raw/*.parquet data/interim/*.parquet data/processed/*.parquet
	@echo "Données supprimées (les fichiers DVC .dvc restent)"

clean-models:
	-rm -rf models/best_model/
	@echo "Modèles supprimés"
