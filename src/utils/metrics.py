import numpy as np

def _to_numpy(a):
    return np.asarray(a, dtype=float)

def sharpe_ratio(returns_arith, risk_free_annual=0.0, periods_per_year=252):
    """
    Sharpe ratio sur retours ARITHMÉTIQUES journaliers.
    - returns_arith: array-like de r_t (ex: 0.001 = 0.1%).
    - risk_free_annual: taux sans risque annualisé (ex: 0.02 = 2%/an).
    """
    r = _to_numpy(returns_arith)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    rf_daily = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_daily
    std = excess.std(ddof=0)
    if std == 0:
        return np.nan
    return (excess.mean() / std) * np.sqrt(periods_per_year)

def sortino_ratio(returns_arith, risk_free_annual=0.0, periods_per_year=252, mar_annual=None):
    """
    Sortino ratio sur retours ARITHMÉTIQUES.
    - mar_annual: Minimum Acceptable Return annualisé (si None, utilise risk-free).
    """
    r = _to_numpy(returns_arith)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    if mar_annual is None:
        mar_annual = risk_free_annual
    mar_daily = (1.0 + mar_annual) ** (1.0 / periods_per_year) - 1.0
    downside = np.clip(mar_daily - r, 0.0, None)  # “écart à la baisse”
    dd = downside.std(ddof=0)
    if dd == 0:
        return np.nan
    return ((r.mean() - mar_daily) / dd) * np.sqrt(periods_per_year)

def annualized_volatility(returns_arith, periods_per_year=252):
    r = _to_numpy(returns_arith)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    return r.std(ddof=0) * np.sqrt(periods_per_year)

def max_drawdown(values):
    """
    Max drawdown d'une courbe de valeur (array-like), pas des retours.
    Retourne un nombre négatif (ex: -0.25 = -25%).
    """
    v = _to_numpy(values)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    peak = np.maximum.accumulate(v)
    drawdowns = (v - peak) / (peak + 1e-12)
    return float(drawdowns.min())

def drawdown_series(values):
    """
    Série des drawdowns point par point (même définition que max_drawdown, mais renvoie tout le vecteur).
    """
    v = _to_numpy(values)
    peak = np.maximum.accumulate(v)
    return (v - peak) / (peak + 1e-12)

def annualized_return(values, periods_per_year: int = 252):
    """
    Rendement annualisé (CAGR) d'une courbe de valeur (start -> end).
    """
    v = _to_numpy(values)
    v = v[np.isfinite(v)]
    if v.size < 2 or v[0] <= 0 or v[-1] <= 0:
        return 0.0
    n = v.size
    return (v[-1] / v[0]) ** (periods_per_year / n) - 1.0
