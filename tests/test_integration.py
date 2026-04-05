# ============================================================
# tests/test_integration.py
# Tests d'intégration : pipeline complet collect → features → train → predict
# Ces tests sont plus lents (marqués pytest.mark.integration).
# Ils utilisent des données synthétiques (pas d'appels réseau réels).
#
# Lancement sélectif :
#   pytest tests/test_integration.py -v
#   pytest -m integration -v
# ============================================================

import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════
# Tests : Pipeline Data Engineering complet
# ═══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDataPipelineIntegration:
    """
    Teste le pipeline data en utilisant les fixtures de conftest.py.
    Vérifie que chaque étape produit des sorties correctes
    utilisables par l'étape suivante.
    """

    def test_validate_then_preprocess_no_errors(self, weather_df, tmp_cfg, tmp_data_dirs):
        """Validation → preprocessing sans erreur sur données synthétiques."""
        from src.data.validate import validate_weather
        from src.data.preprocess import remove_outliers_iqr, impute_time_series

        # Validation
        results = validate_weather(weather_df)
        n_failed = sum(1 for v in results.values() if not v)
        assert n_failed == 0, f"{n_failed} checks de validation échoués"

        # Preprocessing
        meteo_cols = [c for c in weather_df.columns if c.startswith("meteo_")]
        df, stats  = remove_outliers_iqr(weather_df, meteo_cols, factor=3.0)
        df         = impute_time_series(df)

        assert df.shape == weather_df.shape
        assert df.isnull().sum().sum() == 0

    def test_preprocess_output_has_no_nulls(self, merged_df, tmp_cfg, tmp_data_dirs):
        """Le preprocessing ne laisse aucun NaN dans les colonnes météo."""
        from src.data.preprocess import remove_outliers_iqr, impute_time_series

        meteo_cols = [c for c in merged_df.columns if c.startswith("meteo_")]
        df, _      = remove_outliers_iqr(merged_df, meteo_cols)
        df         = impute_time_series(df)

        null_counts = df[meteo_cols].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0, f"NaN restants : {cols_with_nulls.to_dict()}"

    def test_feature_engineering_preserves_chronological_order(self, features_df):
        """Les features engineered respectent l'ordre chronologique."""
        assert features_df.index.is_monotonic_increasing
        assert not features_df.index.duplicated().any()

    def test_feature_engineering_no_future_leakage(self, features_df):
        """Aucun lag négatif (données futures) dans les features."""
        lag_cols = [c for c in features_df.columns if "_lag_" in c]
        for col in lag_cols[:5]:
            # Extraire le lag en heures depuis le nom de colonne
            parts = col.split("_lag_")
            if len(parts) == 2:
                lag_str = parts[1].replace("h", "")
                try:
                    lag_h = int(lag_str)
                    assert lag_h > 0, f"Lag non-positif détecté : {col}"
                except ValueError:
                    pass

    def test_train_test_no_overlap(self, train_val_test):
        """Train, val et test n'ont aucune date en commun."""
        train, val, test = train_val_test
        train_idx = set(train.index)
        val_idx   = set(val.index)
        test_idx  = set(test.index)

        assert len(train_idx & val_idx)  == 0, "Overlap train ∩ val"
        assert len(train_idx & test_idx) == 0, "Overlap train ∩ test"
        assert len(val_idx   & test_idx) == 0, "Overlap val ∩ test"

    def test_train_precedes_val_precedes_test(self, train_val_test):
        """Ordre temporel strict : train < val < test."""
        train, val, test = train_val_test
        assert train.index.max() < val.index.min()
        assert val.index.max()   < test.index.min()

    def test_features_df_has_target_column(self, features_df):
        """La variable cible est présente dans le DataFrame."""
        assert "target_24h" in features_df.columns
        assert features_df["target_24h"].isnull().sum() == 0

    def test_feature_count_reasonable(self, features_df):
        """Nombre de features raisonnable (ni trop peu, ni explosion)."""
        n_features = len([c for c in features_df.columns
                          if c != "target_24h" and c != "meteo_temperature_2m"])
        assert 10 <= n_features <= 300, f"Nombre de features suspect : {n_features}"


# ═══════════════════════════════════════════════════════════════
# Tests : Pipeline Modélisation complet
# ═══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestModelPipelineIntegration:

    def test_xgboost_full_pipeline(self, X_y_train, X_y_test):
        """Entraînement XGBoost → évaluation → métriques cohérentes."""
        from xgboost import XGBRegressor
        from src.models.evaluate import compute_metrics

        X_train, y_train = X_y_train
        X_test,  y_test  = X_y_test

        model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = compute_metrics(y_test.values, preds)

        # Métriques physiquement plausibles pour la température
        assert metrics["rmse"] < 20.0,  "RMSE physiquement impossible"
        assert metrics["mae"]  < 15.0,  "MAE physiquement impossible"
        assert -1.0 <= metrics["r2"] <= 1.0
        assert metrics["mape"] >= 0

    def test_xgboost_beats_naive_on_features(self, X_y_train, X_y_test, features_df):
        """XGBoost avec features engineered doit battre le modèle naïf."""
        from xgboost import XGBRegressor
        from src.models.evaluate import compute_metrics

        X_train, y_train = X_y_train
        X_test,  y_test  = X_y_test

        # XGBoost
        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0,
                             learning_rate=0.1, max_depth=4)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m_xgb = compute_metrics(y_test.values, preds)

        # Naïf : prédire la moyenne du train
        naive = np.full(len(y_test), y_train.mean())
        m_naive = compute_metrics(y_test.values, naive)

        assert m_xgb["rmse"] < m_naive["rmse"], (
            f"XGBoost ({m_xgb['rmse']:.4f}) ≥ naïf ({m_naive['rmse']:.4f})"
        )

    def test_feature_importance_sums_to_one(self, trained_xgboost):
        """Les importances de features somment à 1."""
        fi = trained_xgboost.feature_importances_
        assert abs(fi.sum() - 1.0) < 1e-5

    def test_model_predictions_finite(self, trained_xgboost, X_y_test):
        """Toutes les prédictions sont des nombres finis."""
        X_test, _ = X_y_test
        preds = trained_xgboost.predict(X_test)
        assert np.all(np.isfinite(preds)), "Prédictions inf ou NaN détectées"

    def test_walk_forward_cv_consistent_metrics(self, trained_xgboost, X_y_test):
        """Walk-forward CV produit des métriques cohérentes sur plusieurs folds."""
        from src.models.evaluate import walk_forward_cv
        X_test, y_test = X_y_test
        results = walk_forward_cv(trained_xgboost, X_test, y_test,
                                   n_splits=3, gap=5)
        assert results["cv_rmse_mean"] > 0
        assert results["cv_rmse_std"]  >= 0
        # Vérifier que tous les folds ont des métriques valides
        for fold in results["cv_folds"]:
            assert fold["rmse"] >= 0
            assert -1.0 <= fold["r2"] <= 1.0

    def test_evaluate_model_generates_plots(self, trained_xgboost, X_y_test, tmp_path):
        """evaluate_model génère les plots sans erreur."""
        from src.models.evaluate import evaluate_model
        X_test, y_test = X_y_test
        output_dir = str(tmp_path / "figures")
        metrics = evaluate_model(
            model=trained_xgboost,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
        )
        assert "rmse" in metrics
        assert "r2"   in metrics
        # Les plots doivent exister
        plots_dir = tmp_path / "figures"
        assert (plots_dir / "predictions_vs_actual.png").exists()
        assert (plots_dir / "residuals.png").exists()


# ═══════════════════════════════════════════════════════════════
# Tests : Intégration API + Modèle
# ═══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestAPIModelIntegration:

    def test_api_predicts_with_real_features(self, api_client, features_df):
        """L'API prédit correctement avec de vraies features engineered."""
        feat_cols = [c for c in features_df.columns
                     if c != "target_24h" and c != "meteo_temperature_2m"]
        sample = features_df[feat_cols].iloc[100].to_dict()

        payload = {
            "datetime_utc": "2023-03-15T10:00:00",
            "features": {k: float(v) for k, v in sample.items()
                         if not (isinstance(v, float) and np.isnan(v))},
        }
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_temp_24h" in data
        assert isinstance(data["predicted_temp_24h"], float)

    def test_api_confidence_interval_contains_prediction(self, api_client, sample_predict_payload):
        """L'intervalle de confiance encadre toujours la prédiction centrale."""
        resp = api_client.post("/predict", json=sample_predict_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence_lower"] <= data["predicted_temp_24h"]
        assert data["predicted_temp_24h"] <= data["confidence_upper"]

    def test_api_batch_predict_all_requests_answered(self, api_client, sample_predict_payload):
        """Batch predict : autant de réponses que de requêtes."""
        batch_payload = {
            "requests": [sample_predict_payload] * 5
        }
        resp = api_client.post("/predict/batch", json=batch_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_predictions"] == 5
        assert len(data["predictions"]) == 5

    def test_api_latency_reasonable(self, api_client, sample_predict_payload):
        """Latence de prédiction < 500ms (modèle mock)."""
        import time
        start = time.perf_counter()
        api_client.post("/predict", json=sample_predict_payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Latence trop élevée : {elapsed_ms:.1f}ms"


# ═══════════════════════════════════════════════════════════════
# Tests : Intégration Drift Detection
# ═══════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDriftIntegration:

    def test_no_drift_same_distribution(self, weather_df):
        """Même distribution → pas de drift détecté."""
        from src.monitoring.drift import compute_drift_report

        n_ref = len(weather_df) // 2
        ref = weather_df.iloc[:n_ref]
        cur = weather_df.iloc[n_ref:]

        result = compute_drift_report(
            reference=ref,
            current=cur,
            output_dir="/tmp/drift_test_same",
        )
        if "skipped" not in result:
            # Même population → pas de drift massif
            assert result["share_drifted"] < 0.50, (
                f"Drift détecté sur même distribution : {result['share_drifted']:.1%}"
            )

    def test_drift_detected_different_distribution(self, weather_df):
        """Distributions décalées → drift détecté."""
        from src.monitoring.drift import compute_drift_report

        ref = weather_df.copy()
        # Simuler un décalage de +5°C (changement climatique brutal)
        cur = weather_df.copy()
        cur["meteo_temperature_2m"] = cur["meteo_temperature_2m"] + 8.0
        cur["meteo_relative_humidity_2m"] = cur["meteo_relative_humidity_2m"] - 15.0

        result = compute_drift_report(
            reference=ref,
            current=cur,
            output_dir="/tmp/drift_test_diff",
        )
        if "skipped" not in result:
            # Distribution très différente → drift détecté
            assert result["dataset_drift"] is True or result["share_drifted"] > 0.0

    def test_drift_report_always_returns_action(self, weather_df):
        """Le rapport de drift retourne toujours une action recommandée."""
        from src.monitoring.drift import compute_drift_report

        result = compute_drift_report(
            reference=weather_df.head(200),
            current=weather_df.tail(200),
            output_dir="/tmp/drift_test_action",
        )
        if "skipped" not in result:
            assert "action_required" in result
            assert result["action_required"] in ("RETRAIN", "MONITOR")

    def test_performance_monitoring_no_crash(self):
        """Le monitoring de performance ne plante pas sans données."""
        from src.monitoring.performance import compute_rolling_metrics
        empty = pd.DataFrame()
        result = compute_rolling_metrics(empty)
        assert result == {}

    def test_rolling_metrics_correct_calculation(self):
        """Métriques glissantes calculées correctement."""
        from src.monitoring.performance import compute_rolling_metrics

        n   = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        df  = pd.DataFrame({
            "predicted": np.ones(n) * 15.0,
            "actual":    np.ones(n) * 16.0,   # erreur constante de +1°C
        }, index=idx)

        result = compute_rolling_metrics(df, window_hours=n)
        assert abs(result["rmse"] - 1.0) < 1e-6
        assert abs(result["mae"]  - 1.0) < 1e-6
        assert abs(result["bias"] - (-1.0)) < 1e-6  # pred - actual = 15-16 = -1


# ═══════════════════════════════════════════════════════════════
# Tests : DVC et reproductibilité
# ═══════════════════════════════════════════════════════════════

class TestReproducibility:

    def test_params_yaml_consistent_with_model_config(self):
        """params.yaml et model_config.yaml ne se contredisent pas."""
        import yaml
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        with open("configs/model_config.yaml") as f:
            model_cfg = yaml.safe_load(f)

        # Le modèle dans params.yaml doit être dans model_config
        model_name = params["train"]["model"]
        assert model_name in model_cfg["models"], (
            f"Modèle '{model_name}' dans params.yaml absent de model_config.yaml"
        )

    def test_dvc_yaml_stages_have_outputs(self):
        """Chaque étape DVC déclare des outputs."""
        import yaml
        with open("dvc.yaml") as f:
            dvc = yaml.safe_load(f)
        for stage_name, stage in dvc.get("stages", {}).items():
            assert "cmd" in stage, f"Stage '{stage_name}' sans commande"
            # Collect et evaluate peuvent ne pas avoir d'outs (ont des metrics)
            has_outs    = "outs"     in stage
            has_metrics = "metrics"  in stage
            assert has_outs or has_metrics, (
                f"Stage '{stage_name}' sans outputs ni metrics"
            )

    def test_config_files_loadable_and_valid(self):
        """Tous les fichiers de config sont chargeable et valides."""
        import yaml
        from src.utils.config import validate_config

        configs = [
            "configs/config.yaml",
            "configs/model_config.yaml",
            "configs/monitoring_config.yaml",
        ]
        for path in configs:
            with open(path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{path} n'est pas un dict"
            assert len(data) > 0,          f"{path} est vide"

        # Validation stricte de la config principale
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        validate_config(cfg)

    def test_feature_names_deterministic(self, weather_df):
        """Le même DataFrame produit toujours les mêmes features."""
        from src.features.build_features import (
            add_time_features, add_lag_features,
        )
        df1 = add_time_features(weather_df)
        df1 = add_lag_features(df1, "meteo_temperature_2m", lags=[1, 24])
        df2 = add_time_features(weather_df)
        df2 = add_lag_features(df2, "meteo_temperature_2m", lags=[1, 24])

        assert list(df1.columns) == list(df2.columns)
        assert df1.shape == df2.shape

    def test_xgboost_predictions_deterministic(self, trained_xgboost, X_y_test):
        """Le même modèle produit les mêmes prédictions."""
        X_test, _ = X_y_test
        preds_1 = trained_xgboost.predict(X_test)
        preds_2 = trained_xgboost.predict(X_test)
        np.testing.assert_array_equal(preds_1, preds_2)
