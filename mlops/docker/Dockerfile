# ============================================================
# Dockerfile.api — Image de production pour l'API FastAPI
#
# Build : docker build -f mlops/docker/Dockerfile.api -t ts-forecast-api:latest .
# Run   : docker run -p 8000:8000 ts-forecast-api:latest
# ============================================================

# ─── Stage 1 : Builder (installe les dépendances) ────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Variables build-time
ARG BUILD_DATE
ARG GIT_COMMIT
LABEL build_date="${BUILD_DATE}" git_commit="${GIT_COMMIT}"

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement les requirements pour profiter du cache Docker
COPY requirements.txt .

# Installer dans un répertoire isolé pour copie en stage 2
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2 : Runtime (image finale légère) ──────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# Métadonnées
LABEL maintainer="mlops-team"
LABEL description="ts-forecast API — Prévision séries temporelles météo"
LABEL version="1.0.0"

# Copier les packages installés depuis builder
COPY --from=builder /install /usr/local

# Utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Créer les répertoires nécessaires
RUN mkdir -p /app/logs /app/models /app/reports/monitoring && \
    chown -R appuser:appuser /app

# Copier le code applicatif
COPY --chown=appuser:appuser src/      ./src/
COPY --chown=appuser:appuser api/      ./api/
COPY --chown=appuser:appuser configs/  ./configs/

# Passer en utilisateur non-root
USER appuser

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000

# Port exposé
EXPOSE 8000

# Health check Docker natif
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Commande de démarrage
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--access-log", \
     "--log-level", "info"]
