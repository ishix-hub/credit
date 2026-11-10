"""Data loading and statistical helpers for the dashboard."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).parent / "DATA"

REGIMES = [
    ("1971-72 to 1990-91 | Pre-reform", 1971, 1990),
    ("1991-92 to 2003-04 | Reform years", 1991, 2003),
    ("2004-05 to 2010-11 | Credit boom", 2004, 2010),
    ("2011-12 to 2019-20 | Bank stress", 2011, 2019),
    ("2020-21 to 2025-26 | Covid and revival", 2020, 2025),
]

EPISODES = {
    "Credit-led boom (2004-05 to 2007-08)": (2004, 2007,
        "Credit growth ran far above nominal GDP growth, pushing credit-to-GDP up sharply while GDP growth was at its highest in the sample."),
    "Global financial crisis (2008-09 to 2009-10)": (2008, 2009,
        "GDP growth dropped with the external shock; credit growth slowed more gradually, which is what a lagging credit cycle looks like."),
    "Post-crisis rebound (2010-11)": (2010, 2010,
        "A strong recovery year for both credit and output."),
    "Slowdown and stress (2011-12 to 2013-14)": (2011, 2013,
        "Credit growth fell steadily as bank balance sheets weakened and GDP growth slowed."),
    "Credit slump and asset-quality clean-up (2014-15 to 2017-18)": (2014, 2017,
        "Credit growth fell below 10% while real GDP growth stayed high, so a credit slump did not produce a growth slump. Gross NPAs peaked in 2017-18."),
    "NBFC stress and pre-Covid slowdown (2018-19 to 2019-20)": (2018, 2019,
        "Tighter funding for non-bank lenders coincided with a slowdown in GDP growth."),
    "Covid-19 (2020-21)": (2020, 2020,
        "GDP contracted sharply. Credit growth was the lowest in the sample, but its fall was much smaller than the swing in output."),
    "Revival (2021-22 to 2025-26)": (2021, 2025,
        "Credit growth has run well above nominal GDP growth, lifting credit-to-GDP and the credit-to-deposit ratio to new highs."),
}


def load_annual() -> pd.DataFrame:
    a = pd.read_csv(DATA / "annual_credit_gdp.csv")
    a["fy"] = a["year"].str[:4].astype(int)
    a["regime"] = "Other"
    for name, lo, hi in REGIMES:
        a.loc[(a.fy >= lo) & (a.fy <= hi), "regime"] = name
    # real credit growth, deflated by WPI or CPI (independent of GDP)
    for tag, col in (("wpi", "wpi_inflation"), ("cpi", "cpi_inflation")):
        a[f"real_credit_growth_{tag}"] = ((1 + a.credit_growth / 100) / (1 + a[col] / 100) - 1) * 100
    a["credit_gdp_gap"] = a.credit_growth - a.nominal_gdp_growth
    a["credit_to_gdp_change"] = a.credit_to_gdp.diff()
    return a


def real_levels(a: pd.DataFrame, deflator: str) -> pd.DataFrame:
    """Log real credit and log real GDP index on the years where the deflator exists."""
    col = {"wpi": "wpi_inflation", "cpi": "cpi_inflation"}[deflator]
    d = a.dropna(subset=[col]).copy()
    d["price_index"] = (1 + d[col] / 100).cumprod()
    d["real_credit"] = d.credit_outstanding_cr / d.price_index
    d["real_gdp"] = (1 + d.real_gdp_growth / 100).cumprod()
    d["ln_credit"] = np.log(d.real_credit)
    d["ln_gdp"] = np.log(d.real_gdp)
    return d


def hp_filter(y: np.ndarray, lam: float = 6.25):
    """Hodrick-Prescott filter; returns (trend, cycle)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    d2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        d2[i, i:i + 3] = [1, -2, 1]
    trend = np.linalg.solve(np.eye(n) + lam * d2.T @ d2, y)
    return trend, y - trend


def pearson(x, y):
    m = pd.concat([pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1).dropna()
    if len(m) < 4:
        return dict(n=len(m), r=np.nan, p=np.nan, rho=np.nan, prho=np.nan, slope=np.nan, intercept=np.nan)
    r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    rho, prho = stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])
    s = stats.linregress(m.iloc[:, 0], m.iloc[:, 1])
    return dict(n=len(m), r=r, p=p, rho=rho, prho=prho, slope=s.slope, intercept=s.intercept)


def cross_corr(credit: pd.Series, gdp: pd.Series, kmax: int) -> pd.DataFrame:
    """corr(credit[t-k], gdp[t]). k > 0: credit leads GDP. k < 0: credit lags GDP."""
    rows = []
    for k in range(-kmax, kmax + 1):
        x = credit.shift(k)
        m = pd.concat([x, gdp], axis=1).dropna()
        if len(m) > 4:
            r, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
            rows.append((k, r, p, len(m)))
    return pd.DataFrame(rows, columns=["k", "r", "p", "n"])


def granger(y: pd.Series, x: pd.Series, p: int):
    """F-test that lags of x help predict y (x Granger-causes y)."""
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    cols = {"y": d.y}
    for i in range(1, p + 1):
        cols[f"y{i}"] = d.y.shift(i)
        cols[f"x{i}"] = d.x.shift(i)
    z = pd.DataFrame(cols).dropna()
    if len(z) < 2 * p + 6:
        return np.nan, np.nan, len(z)
    Y = z.y.values
    Xr = np.column_stack([np.ones(len(z))] + [z[f"y{i}"] for i in range(1, p + 1)])
    Xu = np.column_stack([Xr] + [z[f"x{i}"] for i in range(1, p + 1)])
    rss = lambda X: float(((Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]) ** 2).sum())
    rr, ru = rss(Xr), rss(Xu)
    df2 = len(z) - Xu.shape[1]
    f = ((rr - ru) / p) / (ru / df2)
    return f, float(stats.f.sf(f, p, df2)), len(z)


def fit_ols(y, X):
    """OLS with intercept already in X. Returns coefficients, residuals, sigma."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    sigma = np.sqrt((res ** 2).sum() / (len(y) - X.shape[1]))
    return beta, res, sigma


def cagr(a, b, years):
    return ((b / a) ** (1 / years) - 1) * 100
