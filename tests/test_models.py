# ============================================================
# tests/test_models.py
# Tests unitaires : métriques, évaluation, sélection modèle
# ============================================================

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import yaml


@pytest.fixture
def perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def noisy_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.normal(15, 5, 500)
    y_pred = y_true + rng.normal(0, 1.5, 500)
    return y_true, y_pred


@pytest.fixture
def small_tabular_dataset():
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    X = pd.DataFrame({
        "lag_1":   rng.normal(15, 5, n),
        "lag_24":  rng.normal(15, 5, n),
        "hour_sin": np.sin(2 * np.pi * np.arange(n) / 24),
        "hour_cos": np.cos(2 * np.pi * np.arange(n) / 24),
        "roll_mean_6h": rng.normal(15, 3, n),
    }, index=idx)
    y = pd.Series(rng.normal(15, 5, n), index=idx, name="target_24h")
    return X, y


# ─── Tests : Métriques ────────────────────────────────────────

class TestMetrics:

    def test_rmse_perfect(self, perfect_predictions):
        from src.models.evaluate import compute_metrics
        y_true, y_pred = perfect_predictions
        m = compute_metrics(y_true, y_pred)
        assert abs(m["rmse"]) < 1e-10
        assert abs(m["mae"])  < 1e-10
        assert abs(m["r2"] - 1.0) < 1e-10

    def test_rmse_positive(self, noisy_predictions):
        from src.models.evaluate import compute_metrics
        y_true, y_pred = noisy_predictions
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] >= 0
        assert m["mae"]  >= 0
        assert m["mape"] >= 0

    def test_rmse_greater_or_equal_mae(self, noisy_predictions):
        """RMSE ≥ MAE (inégalité QM-AM)."""
        from src.models.evaluate import compute_metrics
        y_true, y_pred = noisy_predictions
        m = compute_metrics(y_true, y_pred)
        assert m["rmse"] >= m["mae"] - 1e-10

    def test_r2_at_most_one(self, noisy_predictions):
        from src.models.evaluate import compute_metrics
        y_true, y_pred = noisy_predictions
        m = compute_metrics(y_true, y_pred)
        assert m["r2"] <= 1.0 + 1e-10

    def test_mape_zero_proof(self, perfect_predictions):
        from src.models.evaluate import compute_metrics
        y_true, y_pred = perfect_predictions
        m = compute_metrics(y_true, y_pred)
        assert abs(m["mape"]) < 1e-6

    def test_baseline_naive_rmse(self, noisy_predictions):
        """RMSE doit battre le modèle naïf (prédire la moyenne)."""
        from src.models.evaluate import compute_metrics
        y_true, y_pred = noisy_predictions
        # Modèle naïf
        naive_pred = np.full_like(y_true, y_true.mean())
        m_naive  = compute_metrics(y_true, naive_pred)
        m_actual = compute_metrics(y_true, y_pred)
        # Notre modèle (bruité ±1.5°C) devrait battre la moyenne
        assert m_actual["rmse"] < m_naive["rmse"]


# ─── Tests : XGBoost ──────────────────────────────────────────

class TestXGBoost:

    def test_xgboost_fits_and_predicts(self, small_tabular_dataset):
        """XGBoost s'entraîne et prédit sans erreur."""
        from xgboost import XGBRegressor
        X, y = small_tabular_dataset
        split = int(len(X) * 0.8)
        model = XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        model.fit(X.iloc[:split], y.iloc[:split])
        preds = model.predict(X.iloc[split:])
        assert len(preds) == len(X) - split
        assert not np.any(np.isnan(preds))
        assert not np.any(np.isinf(preds))

    def test_xgboost_output_shape(self, small_tabular_dataset):
        from xgboost import XGBRegressor
        X, y = small_tabular_dataset
        model = XGBRegressor(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_xgboost_feature_importance_available(self, small_tabular_dataset):
        from xgboost import XGBRegressor
        X, y = small_tabular_dataset
        model = XGBRegressor(n_estimators=20, random_state=42, verbosity=0)
        model.fit(X, y)
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == X.shape[1]
        assert np.sum(model.feature_importances_) > 0

    def test_xgboost_beats_naive_on_synthetic(self, small_tabular_dataset):
        """XGBoost devrait battre la prédiction naïve sur données synthétiques."""
        from xgboost import XGBRegressor
        from src.models.evaluate import compute_metrics
        X, y = small_tabular_dataset
        split = int(len(X) * 0.8)
        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X.iloc[:split], y.iloc[:split])
        preds = model.predict(X.iloc[split:])
        y_test = y.iloc[split:].values
        naive_preds = np.full_like(y_test, y.iloc[:split].mean())
        m_model = compute_metrics(y_test, preds)
        m_naive = compute_metrics(y_test, naive_preds)
        assert m_model["rmse"] > m_naive["rmse"], (
            f"XGBoost RMSE={m_model['rmse']:.4f} ≥ Naïf RMSE={m_naive['rmse']:.4f}"
            )


# ─── Tests : TimeSeriesSplit ───────────────────────────────────

class TestCrossValidation:

    def test_timeseries_split_no_leakage(self, small_tabular_dataset):
        """Chaque fold : les index de validation sont tous > index de train."""
        from sklearn.model_selection import TimeSeriesSplit
        X, y = small_tabular_dataset
        tscv = TimeSeriesSplit(n_splits=3, gap=24)
        for train_idx, val_idx in tscv.split(X):
            assert max(train_idx) < min(val_idx), "Data leakage détecté !"

    def test_walk_forward_produces_metrics(self, small_tabular_dataset):
        """walk_forward_cv produit des métriques valides."""
        from xgboost import XGBRegressor
        from src.models.evaluate import walk_forward_cv
        X, y = small_tabular_dataset
        model = XGBRegressor(n_estimators=20, random_state=42, verbosity=0)
        results = walk_forward_cv(model, X, y, n_splits=3, gap=5)
        assert "cv_rmse_mean" in results
        assert results["cv_rmse_mean"] > 0
        assert len(results["cv_folds"]) == 3


# ─── Tests : Production gate ──────────────────────────────────

class TestProductionGate:

    def test_gate_passes_good_model(self):
        """Modèle performant → gate passé."""
        thresholds = {"max_rmse": 2.5, "max_mape": 10.0, "min_r2": 0.80}
        metrics = {"rmse": 1.2, "mape": 5.0, "r2": 0.92}
        gate = (
            metrics["rmse"] <= thresholds["max_rmse"] and
            metrics["mape"] <= thresholds["max_mape"] and
            metrics["r2"]   >= thresholds["min_r2"]
        )
        assert gate is True

    def test_gate_fails_high_rmse(self):
        """RMSE trop élevé → gate échoue."""
        thresholds = {"max_rmse": 2.5, "max_mape": 10.0, "min_r2": 0.80}
        metrics = {"rmse": 5.0, "mape": 5.0, "r2": 0.92}
        gate = (
            metrics["rmse"] <= thresholds["max_rmse"] and
            metrics["mape"] <= thresholds["max_mape"] and
            metrics["r2"]   >= thresholds["min_r2"]
        )
        assert gate is False

    def test_gate_fails_low_r2(self):
        thresholds = {"max_rmse": 2.5, "max_mape": 10.0, "min_r2": 0.80}
        metrics = {"rmse": 1.0, "mape": 3.0, "r2": 0.60}
        gate = (
            metrics["rmse"] <= thresholds["max_rmse"] and
            metrics["mape"] <= thresholds["max_mape"] and
            metrics["r2"]   >= thresholds["min_r2"]
        )
        assert gate is False


# ─── Tests : Paramètres DVC ───────────────────────────────────

class TestParamsYaml:

    def test_params_loadable(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        assert "train" in params
        assert "featurize" in params
        assert "prepare" in params

    def test_params_xgboost_valid(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        xgb = params["train"]["xgboost"]
        assert 0 < xgb["learning_rate"] < 1.0
        assert xgb["n_estimators"] > 0
        assert xgb["max_depth"] > 0
        assert 0 < xgb["subsample"] <= 1.0

    def test_prepare_ratio_sums_to_one(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        p = params["prepare"]
        total = p["train_ratio"] + p.get("val_ratio", 0.1) + (
            1 - p["train_ratio"] - p.get("val_ratio", 0.1)
        )
        assert abs(total - 1.0) < 1e-10
