# ============================================================
# src/analysis/statistical_tests.py
# Tests statistiques complets pour séries temporelles :
#   - ADF + KPSS (stationnarité)
#   - Décomposition STL (Décomposition saisonnière  + tendance + résidus)
#   - ACF / PACF 
#   - Test de Granger (si une série permet de prédire une autre)
#   - Ljung-Box (autocorrélation résidus)
#   - Test de Shapiro-Wilk (normalité résidus)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from statsmodels.tsa.stattools  import adfuller, kpss, grangercausalitytests, acf, pacf
from statsmodels.tsa.seasonal   import STL
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

TARGET = "meteo_temperature_2m"

# --- stationnarité ---
def test_stationarity(series: pd.Series, name: str = "") -> dict:
    """
    ADF (H0 : racine unitaire → non-stationnaire)
    + KPSS (H0 : stationnaire)
    Interprétation conjointe pour diagnostiquer le type de non-stationnarité.
    """
    label = name or series.name
    clean = series.dropna()
    print(f"\n{'═'*55}")
    print(f"  Test stationnarité : {label}")
    print(f"{'═'*55}")

    # ADF
    # H0 : racine unitaire → non-stationnaire
    adf_stat, adf_p, n_lags, n_obs, crit, icbest = adfuller(clean, autolag="AIC")
    adf_stat_stationry = adf_p < 0.05
    print(f"\n  ADF  | stat={adf_stat:.4f} | p={adf_p:.4f} | lags={n_lags}")
    for level, val in crit.items():
        flag = "OK!!!" if adf_stat < val else "  "
        print(f"Critique {level}: {val:.4f} {flag}")

    # KPSS
    # H0 : stationnaire
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(clean, regression="c", nlags="auto")
    kpss_stationary = kpss_p > 0.05
    print(f"\n  KPSS | stat={kpss_stat:.4f} | p={kpss_p:.4f} | lags={kpss_lags}")
    print(f"  ➜ {'Stationnaire' if kpss_stationary else 'NON stationnaire'} (α=5%)")

    # Interprétation conjointe
    if adf_stat_stationry and kpss_stationary:
        conclusion = "STATIONNAIRE"
    elif not adf_stat_stationry and not kpss_stationary:
        conclusion = "NON_STATIONNAIRE"
    elif adf_stat_stationry and not kpss_stationary:
        conclusion = "TREND_STATIONARY"     # tendance déterministe
    else:
        conclusion = "DIFFERENCE_STATIONARY"  # racine unitaire stochastique

    print(f"\nConclusion : {conclusion}")

    return {
        "series_name":      label,
        "adf_stat":         float(adf_stat),
        "adf_pval":         float(adf_p),
        "kpss_stat":        float(kpss_stat),
        "kpss_pval":        float(kpss_p),
        "stationary_adf":   bool(adf_stat_stationry),
        "stationary_kpss":  bool(kpss_stationary),
        "conclusion":       conclusion,
        "differencing_recommended": conclusion in ("NON_STATIONNAIRE", "DIFFERENCE_STATIONARY"),
    }

# Décomposition STL
def decompose_stl(series: pd.Series, period: int = 24,
                  output_dir: str = "reports/figures") -> dict:
    """
    Seasonal-Trend décomposition par Loess (STL).
    Robuste aux outliers. Période = 24 pour saisonnalité journalière.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clean = series.dropna()

    stl    = STL(clean, period=period, robust=True, seasonal=7)
    result = stl.fit()

    # Force de saisonnalité et de tendance (Wang et al. 2006)
    var_residual  = result.resid.var()
    var_trend_res = (result.trend + result.resid).var()
    var_seas_res  = (result.seasonal + result.resid).var()
    fs = max(0.0, 1 - var_residual / var_seas_res)   # Force saisonnalité
    ft = max(0.0, 1 - var_residual / var_trend_res)  # Force tendance

    print(f"\n  STL Décomposition (période={period}h)")
    print(f"    Force saisonnalité : {fs:.4f}  {'[FORTE]' if fs > 0.6 else '[faible]'}")
    print(f"    Force tendance     : {ft:.4f}  {'[FORTE]' if ft > 0.6 else '[faible]'}")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    result.observed.plot(ax=axes[0], color="#185FA5", linewidth=0.8)
    axes[0].set_title("Série originale", fontsize=11)
    result.trend.plot(ax=axes[1], color="#0F6E56", linewidth=1.5)
    axes[1].set_title("Tendance (Loess)", fontsize=11)
    result.seasonal.plot(ax=axes[2], color="#993556", linewidth=0.8)
    axes[2].set_title(f"Saisonnalité (période={period}h)", fontsize=11)
    result.resid.plot(ax=axes[3], color="#888780", linewidth=0.8, alpha=0.8)
    axes[3].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[3].set_title("Résidus", fontsize=11)
    plt.suptitle(f"Décomposition STL — {series.name}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stl_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "period":             period,
        "seasonal_strength":  float(fs),
        "trend_strength":     float(ft),
        "strong_seasonality": fs > 0.6,
        "strong_trend":       ft > 0.6,
    }

# ACF / PACF
def plot_acf_pacf(series: pd.Series, lags: int = 72,
                  output_dir: str = "reports/figures") -> dict:
    """
    ACF  → ordre MA (q) : dernier lag significatif
    PACF → ordre AR (p) : dernier lag significatif
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clean = series.dropna()

    acf_vals  = acf(clean,  nlags=lags, alpha=0.05, fft=True)
    pacf_vals = pacf(clean, nlags=lags, alpha=0.05)

    conf_level = 1.96 / np.sqrt(len(clean))

    # Suggestions d'ordres
    sig_acf  = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) < conf_level]
    sig_pacf = [i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) < conf_level]
    suggested_q = sig_acf[-1]  if sig_acf  else 0
    suggested_p = sig_pacf[-1] if sig_pacf else 0

    print(f"\n  ACF/PACF | Lags significatifs ACF : {sig_acf[:5]}...")
    print(f"            Lags significatifs PACF: {sig_pacf[:5]}...")
    print(f"  Suggestion : p={suggested_p}, q={suggested_q} (ARIMA)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    plot_acf(clean,  lags=lags, ax=ax1, color="#185FA5",
             title=f"ACF — ordre MA (q suggéré: {suggested_q})")
    plot_pacf(clean, lags=lags, ax=ax2, color="#993556",
              title=f"PACF — ordre AR (p suggéré: {suggested_p})")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/acf_pacf.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "suggested_p": int(suggested_p),
        "suggested_q": int(suggested_q),
        "n_significant_acf":  len(sig_acf),
        "n_significant_pacf": len(sig_pacf),
    }

# Test de Granger
def granger_causality(df: pd.DataFrame, target: str,
                      candidates: list, max_lag: int = 24) -> dict:
    """
    H0 : X ne cause pas Y au sens de Granger.
    p < 0.05 → X apporte une information prédictive sur Y.
    """
    print(f"\n  Causalité de Granger ({target})")
    print(f"  {'─'*45}")
    results = {}

    for col in candidates:
        try:
            data = df[[target, col]].dropna()
            if len(data) < max_lag * 4:
                continue
            test = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            # Minimum p-value sur tous les lags
            min_p = min(test[lag][0]["ssr_ftest"][1] for lag in test)
            best_lag = min(test, key=lambda l: test[l][0]["ssr_ftest"][1])
            results[col] = {
                "min_pval":      float(min_p),
                "best_lag":      int(best_lag),
                "causes_target": bool(min_p < 0.05),
            }
            sign = "OK!!!" if min_p < 0.05 else "  "
            print(f"{sign} {col:<45} p={min_p:.4f}  (lag={best_lag})")
        except Exception as e:
            print(f"{col} : {e}")

    n_causes = sum(1 for v in results.values() if v["causes_target"])
    print(f"\n {n_causes}/{len(results)} variables causent {target}")
    return results

# Tests sur les résidus
def test_residuals(residuals: pd.Series) -> dict:
    """
    Ljung-Box (autocorrélation) + Shapiro-Wilk (normalité) + Durbin-Watson.
    """
    res = residuals.dropna().values
    print("\n  Tests sur résidus")
    print(f"  {'─'*40}")

    # Ljung-Box
    lb = acorr_ljungbox(res, lags=[10, 20, 40], return_df=True)
    lb_p_min = float(lb["lb_pvalue"].min())
    autocorr_ok = lb_p_min > 0.05
    no_autocorr = "pas d'autocorrélation"
    autocorr_pb = "autocorrélation résiduelle"
    print(f"  Ljung-Box  p_min={lb_p_min:.4f} "
          f"→ {no_autocorr if autocorr_ok else autocorr_pb}")

    # Shapiro-Wilk (sur échantillon si trop grand)
    sample = res[:5000] if len(res) > 5000 else res
    _, sw_p = stats.shapiro(sample)
    normal_ok = sw_p > 0.05
    print(f"  Shapiro-Wilk p={sw_p:.4f} "
          f"→ {'normalité' if normal_ok else 'non-normalité'}")

    # Durbin-Watson
    dw = durbin_watson(res)
    dw_ok = 1.5 < dw < 2.5
    print(f"  Durbin-Watson = {dw:.4f} "
          f"→ {'pas d\'autocorrélation' if dw_ok else 'autocorrélation'}")

    return {
        "ljung_box_pmin":  lb_p_min,
        "no_autocorr":     bool(autocorr_ok),
        "shapiro_pval":    float(sw_p),
        "normal_residuals": bool(normal_ok),
        "durbin_watson":   float(dw),
    }

# Pipeline complet
def run_all_tests(train_path: str,
                  output_dir_figs: str = "reports/figures",
                  output_json:     str = "reports/statistical_tests.json") -> dict:
    print("\n" + "═" * 55)
    print("  TESTS STATISTIQUES — TIME SERIES")
    print("═" * 55)

    df     = pd.read_parquet(train_path)
    series = df[TARGET]

    all_results = {}

    # 1. Stationnarité sur la série brute
    stat = test_stationarity(series, TARGET)
    all_results["stationarity_raw"] = stat

    # 2. Si non-stationnaire → tester la diff
    if stat["differencing_recommended"]:
        print("\nDifférenciation recommandée → test sur diff(1)")
        stat_d1 = test_stationarity(series.diff().dropna(), f"{TARGET}_diff1")
        all_results["stationarity_diff1"] = stat_d1

    # 3. STL (saisonnalité journalière + hebdomadaire)
    stl_daily  = decompose_stl(series, period=24,  output_dir=output_dir_figs)
    stl_weekly = decompose_stl(series, period=168, output_dir=output_dir_figs)
    all_results["stl_daily"]  = stl_daily
    all_results["stl_weekly"] = stl_weekly

    # 4. ACF / PACF sur la diff si non-stationnaire
    analysis_series = series.diff().dropna() if stat["differencing_recommended"] else series
    acf_res = plot_acf_pacf(analysis_series, lags=72, output_dir=output_dir_figs)
    all_results["acf_pacf"] = acf_res

    # 5. Granger : top features vs cible
    candidates = [
        c for c in df.columns
        if c != TARGET
        and not c.startswith("target")
        and not c.startswith(f"{TARGET}_")
        and df[c].notna().sum() > 100
    ][:10]
    if candidates:
        granger = granger_causality(df, TARGET, candidates, max_lag=24)
        all_results["granger_causality"] = granger

    # 6. Résumé des recommandations
    print("\n" + "═" * 55)
    print("  RECOMMANDATIONS")
    print("═" * 55)
    if stat["differencing_recommended"]:
        print(f">>> Appliquer différenciation d'ordre 1 (d=1)")
    else:
        print(f">>> Série stationnaire — d=0")
    print(f">>> Ordre AR suggéré   : p={acf_res['suggested_p']}")
    print(f">>> Ordre MA suggéré   : q={acf_res['suggested_q']}")
    print(f">>> Saisonnalité forte : {stl_daily['strong_seasonality']} (24h)")

    # Sauvegarde JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        return str(obj)

    with open(output_json, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés : {output_json}")
    return all_results


if __name__ == "__main__":
    run_all_tests("data/processed/features_train.parquet")
