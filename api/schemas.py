# ============================================================
# api/schemas.py
# Schémas Pydantic pour l'API FastAPI
# ============================================================

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class ForecastRequest(BaseModel):
    datetime_utc: str = Field(
        ...,
        description="Datetime UTC de la prédiction (format ISO 8601)",
        example="2024-06-01T12:00:00",
    )
    features: dict[str, float] = Field(
        ...,
        description="Features pré-calculées (lags, rolling, encodages cycliques)",
    )

    @field_validator("datetime_utc")
    @classmethod
    def validate_datetime(cls, v):
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Format datetime invalide : {v} (attendu ISO 8601)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "datetime_utc": "2024-06-01T12:00:00",
                "features": {
                    "meteo_temperature_2m_lag_1h": 18.2,
                    "meteo_temperature_2m_lag_24h": 17.5,
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "month_sin": 0.866,
                    "month_cos": 0.5,
                },
            }
        }
    }


class ForecastResponse(BaseModel):
    datetime_utc:       str   = Field(..., description="Datetime de la prédiction")
    predicted_temp_24h: float = Field(..., description="Température prédite T+24h (°C)")
    confidence_lower:   float = Field(..., description="Borne inférieure IC 95% (°C)")
    confidence_upper:   float = Field(..., description="Borne supérieure IC 95% (°C)")
    model_version:      str   = Field(..., description="Version du modèle utilisé")
    prediction_latency_ms: Optional[float] = Field(None, description="Latence (ms)")


class HealthResponse(BaseModel):
    status:       str  = Field(..., description="'ok' ou 'degraded'")
    model_loaded: bool = Field(..., description="Modèle chargé en mémoire")
    timestamp:    str  = Field(..., description="Timestamp UTC de la vérification")
    version:      str  = Field(default="1.0.0")


class BatchForecastRequest(BaseModel):
    """Prédictions batch (plusieurs datetimes)."""
    requests: list[ForecastRequest] = Field(
        ..., min_length=1, max_length=168,
        description="Liste de requêtes (max 168 = 1 semaine horaire)"
    )


class BatchForecastResponse(BaseModel):
    predictions: list[ForecastResponse]
    n_predictions: int
    model_version: str
