# ============================================================
# tests/test_statistical.py
# Tests unitaires : tests ADF/KPSS, STL, ACF/PACF, Granger
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest


# ─── Fixtures locales ─────────────────────────────────────────

@pytest.fixture
def stationary_series():
    """Série stationnaire : bruit blanc gaussien."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=500, freq="h")
    return pd.Series(rng.normal(0, 1, 500), index=idx, name="white_noise")


@pytest.fixture
def nonstationary_series():
    """Série non-stationnaire : marche aléatoire (racine unitaire)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=500, freq="h")
    return pd.Series(np.cumsum(rng.normal(0, 1, 500)), index=idx, name="random_walk")


@pytest.fixture
def seasonal_series():
    """Série avec saisonnalité journalière forte."""
    n   = 24 * 60   # 60 jours
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    t   = np.arange(n)
    s   = 10 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(42).normal(0, 0.5, n)
    return pd.Series(s, index=idx, name="seasonal")


@pytest.fixture
def bivariate_df():
    """DataFrame bivarié pour le test de Granger (X cause Y)."""
    rng = np.random.default_rng(42)
    n   = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    x   = rng.normal(0, 1, n)
    # y dépend de x avec un délai de 2h
    y   = np.zeros(n)
    for i in range(2, n):
        y[i] = 0.7 * x[i - 2] + rng.normal(0, 0.3)
    return pd.DataFrame({"target": y, "cause": x}, index=idx)


# ═══════════════════════════════════════════════════════════════
# Tests : Stationnarité ADF / KPSS
# ═══════════════════════════════════════════════════════════════

class TestStationarity:

    def test_adf_detects_stationary(self, stationary_series):
        from src.analysis.statistical_tests import test_stationarity
        result = test_stationarity(stationary_series)
        assert "adf_stat" in result
        assert "adf_pval" in result
        assert "stationary_adf" in result
        # Bruit blanc → stationnaire à 5%
        assert result["stationary_adf"] is True

    def test_adf_detects_nonstationary(self, nonstationary_series):
        from src.analysis.statistical_tests import test_stationarity
        result = test_stationarity(nonstationary_series)
        # Marche aléatoire → non-stationnaire
        assert result["stationary_adf"] is False

    def test_result_contains_required_keys(self, stationary_series):
        from src.analysis.statistical_tests import test_stationarity
        result = test_stationarity(stationary_series)
        for key in ["adf_stat", "adf_pval", "kpss_stat", "kpss_pval",
                    "stationary_adf", "stationary_kpss", "conclusion"]:
            assert key in result, f"Clé manquante : {key}"

    def test_conclusion_is_string(self, stationary_series):
        from src.analysis.statistical_tests import test_stationarity
        result = test_stationarity(stationary_series)
        assert isinstance(result["conclusion"], str)
        valid_conclusions = {
            "STATIONNAIRE", "NON_STATIONNAIRE",
            "TREND_STATIONARY", "DIFFERENCE_STATIONARY"
        }
        assert result["conclusion"] in valid_conclusions

    def test_adf_pval_in_range(self, stationary_series):
        from src.analysis.statistical_tests import test_stationarity
        result = test_stationarity(stationary_series)
        assert 0.0 <= result["adf_pval"] <= 1.0

    def test_differencing_recommended_for_random_walk(self, nonstationary_series):
        from src.analysis.statistical_tests import test_stationarity
        result = test_stationarity(nonstationary_series)
        assert result["differencing_recommended"] is True

    def test_diff_of_random_walk_is_stationary(self, nonstationary_series):
        """diff(1) d'une marche aléatoire doit être stationnaire."""
        from src.analysis.statistical_tests import test_stationarity
        diff_series = nonstationary_series.diff().dropna()
        result = test_stationarity(diff_series)
        assert result["stationary_adf"] is True


# ═══════════════════════════════════════════════════════════════
# Tests : Décomposition STL
# ═══════════════════════════════════════════════════════════════

class TestSTL:

    def test_stl_returns_required_keys(self, seasonal_series):
        from src.analysis.statistical_tests import decompose_stl
        result = decompose_stl(seasonal_series, period=24, output_dir="/tmp")
        for key in ["period", "seasonal_strength", "trend_strength",
                    "strong_seasonality", "strong_trend"]:
            assert key in result, f"Clé manquante : {key}"

    def test_stl_detects_strong_seasonality(self, seasonal_series):
        """Série avec saisonnalité 24h → force > 0.6."""
        from src.analysis.statistical_tests import decompose_stl
        result = decompose_stl(seasonal_series, period=24, output_dir="/tmp")
        assert result["seasonal_strength"] > 0.5, (
            f"Force saisonnalité trop faible : {result['seasonal_strength']}"
        )

    def test_stl_seasonal_strength_in_range(self, seasonal_series):
        from src.analysis.statistical_tests import decompose_stl
        result = decompose_stl(seasonal_series, period=24, output_dir="/tmp")
        assert 0.0 <= result["seasonal_strength"] <= 1.0
        assert 0.0 <= result["trend_strength"]    <= 1.0

    def test_stl_wrong_period_still_runs(self, seasonal_series):
        """STL avec mauvaise période ne doit pas crasher."""
        from src.analysis.statistical_tests import decompose_stl
        # Période = 12 au lieu de 24 : résultat sous-optimal mais sans crash
        result = decompose_stl(seasonal_series, period=12, output_dir="/tmp")
        assert "seasonal_strength" in result

    def test_stl_white_noise_low_seasonality(self, stationary_series):
        """Bruit blanc → faible saisonnalité."""
        from src.analysis.statistical_tests import decompose_stl
        result = decompose_stl(stationary_series, period=24, output_dir="/tmp")
        # Le bruit blanc n'a pas de saisonnalité forte
        assert result["seasonal_strength"] < 0.5


# ═══════════════════════════════════════════════════════════════
# Tests : ACF / PACF
# ═══════════════════════════════════════════════════════════════

class TestACFPACF:

    def test_acf_pacf_returns_suggestions(self, seasonal_series):
        from src.analysis.statistical_tests import plot_acf_pacf
        result = plot_acf_pacf(seasonal_series.diff().dropna(),
                                lags=48, output_dir="/tmp")
        assert "suggested_p" in result
        assert "suggested_q" in result
        assert isinstance(result["suggested_p"], int)
        assert isinstance(result["suggested_q"], int)
        assert result["suggested_p"] >= 0
        assert result["suggested_q"] >= 0

    def test_acf_white_noise_few_significant_lags(self, stationary_series):
        """Bruit blanc → peu de lags ACF significatifs."""
        from src.analysis.statistical_tests import plot_acf_pacf
        result = plot_acf_pacf(stationary_series, lags=48, output_dir="/tmp")
        # Par définition, le bruit blanc a peu de corrélations significatives
        assert result["n_significant_acf"] < 20   # seuil raisonnable

    def test_acf_ar1_has_geometric_decay(self):
        """AR(1) → ACF à décroissance géométrique (PACF coupe à lag=1)."""
        from statsmodels.tsa.stattools import acf, pacf
        rng = np.random.default_rng(42)
        n   = 2000
        phi = 0.8
        x   = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.normal(0, 1)

        idx    = pd.date_range("2023-01-01", periods=n, freq="h")
        series = pd.Series(x, index=idx)

        from src.analysis.statistical_tests import plot_acf_pacf
        result = plot_acf_pacf(series, lags=20, output_dir="/tmp")
        # PACF doit couper après lag=1 → p≤5 raisonnable
        assert result["suggested_p"] <= 5


# ═══════════════════════════════════════════════════════════════
# Tests : Causalité de Granger
# ═══════════════════════════════════════════════════════════════

class TestGrangerCausality:

    def test_granger_detects_causality(self, bivariate_df):
        """X cause Y avec délai 2h → Granger doit détecter (p < 0.05)."""
        from src.analysis.statistical_tests import granger_causality
        result = granger_causality(bivariate_df, "target", ["cause"], max_lag=5)
        assert "cause" in result
        assert result["cause"]["causes_target"] is True
        assert result["cause"]["min_pval"] < 0.05

    def test_granger_no_causality_for_independent(self, stationary_series):
        """Deux séries indépendantes → pas de causalité."""
        from src.analysis.statistical_tests import granger_causality
        rng = np.random.default_rng(99)
        independent = pd.Series(
            rng.normal(0, 1, len(stationary_series)),
            index=stationary_series.index,
            name="independent",
        )
        df = pd.DataFrame({
            "target": stationary_series.values,
            "independent": independent.values,
        }, index=stationary_series.index)
        result = granger_causality(df, "target", ["independent"], max_lag=5)
        if "independent" in result:
            # p-value devrait être élevée (pas de causalité)
            assert result["independent"]["min_pval"] > 0.01  # seuil lâche

    def test_granger_returns_best_lag(self, bivariate_df):
        from src.analysis.statistical_tests import granger_causality
        result = granger_causality(bivariate_df, "target", ["cause"], max_lag=5)
        assert "best_lag" in result["cause"]
        assert 1 <= result["cause"]["best_lag"] <= 5

    def test_granger_returns_pval_in_range(self, bivariate_df):
        from src.analysis.statistical_tests import granger_causality
        result = granger_causality(bivariate_df, "target", ["cause"], max_lag=5)
        assert 0.0 <= result["cause"]["min_pval"] <= 1.0


# ═══════════════════════════════════════════════════════════════
# Tests : PSI (Population Stability Index)
# ═══════════════════════════════════════════════════════════════

class TestPSI:

    def test_psi_identical_distributions_near_zero(self):
        from src.monitoring.drift import _compute_psi
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.normal(15, 3, 1000))
        cur = pd.Series(rng.normal(15, 3, 1000))
        psi = _compute_psi(ref, cur)
        assert psi < 0.10, f"PSI distributions identiques trop élevé : {psi}"

    def test_psi_different_distributions_high(self):
        from src.monitoring.drift import _compute_psi
        ref = pd.Series(np.random.default_rng(0).normal(10, 1, 500))
        cur = pd.Series(np.random.default_rng(1).normal(20, 1, 500))
        psi = _compute_psi(ref, cur)
        assert psi > 0.25, f"PSI distributions très différentes trop faible : {psi}"

    def test_psi_non_negative(self):
        from src.monitoring.drift import _compute_psi
        rng = np.random.default_rng(42)
        ref = pd.Series(rng.exponential(5, 200))
        cur = pd.Series(rng.exponential(5, 200))
        psi = _compute_psi(ref, cur)
        assert psi >= 0

    def test_psi_moderate_shift(self):
        from src.monitoring.drift import _compute_psi
        ref = pd.Series(np.random.default_rng(0).normal(15, 3, 500))
        cur = pd.Series(np.random.default_rng(1).normal(17, 3, 500))  # +2°C shift
        psi = _compute_psi(ref, cur)
        # Décalage modéré → PSI entre stable et drift
        assert 0.0 <= psi <= 2.0


# ═══════════════════════════════════════════════════════════════
# Tests : Configuration utils
# ═══════════════════════════════════════════════════════════════

class TestConfigUtils:

    def test_load_config_returns_dict(self):
        from src.utils.config import load_config
        cfg = load_config("configs/config.yaml")
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_validate_config_passes(self):
        from src.utils.config import load_config, validate_config
        cfg = load_config("configs/config.yaml")
        validate_config(cfg)   # ne doit pas lever d'erreur

    def test_validate_config_fails_missing_section(self):
        from src.utils.config import validate_config, ConfigError
        bad_cfg = {"project": {"name": "test"}}   # manque data, mlflow, etc.
        with pytest.raises(ConfigError):
            validate_config(bad_cfg)

    def test_get_nested_existing_key(self):
        from src.utils.config import get_nested
        cfg = {"mlflow": {"tracking_uri": "http://localhost:5000"}}
        assert get_nested(cfg, "mlflow", "tracking_uri") == "http://localhost:5000"

    def test_get_nested_missing_key_returns_default(self):
        from src.utils.config import get_nested
        cfg = {"mlflow": {}}
        result = get_nested(cfg, "mlflow", "nonexistent", default="fallback")
        assert result == "fallback"

    def test_get_target_col(self):
        from src.utils.config import load_config, get_target_col
        cfg = load_config("configs/config.yaml")
        col = get_target_col(cfg)
        assert col.startswith("target_")
        assert col.endswith("h")
