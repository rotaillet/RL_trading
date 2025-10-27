import os
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


__all__ = [
    "make_features",
    "load_price_data",
]


def load_price_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    cache_path: Optional[str] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data for the given tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols compatible with yfinance.
    start, end:
        Date range passed to yfinance (inclusive start, exclusive end).
    cache_path:
        Optional path to a CSV cache. If provided and exists, the cache is
        loaded instead of hitting the network.
    force_download:
        If True the cache (if any) is ignored and the data is downloaded again.

    Returns
    -------
    pandas.DataFrame
        MultiIndex dataframe with the OHLCV fields on the first level and the
        ticker symbol on the second level, matching yfinance's output format.
    """

    if cache_path and os.path.exists(cache_path) and not force_download:
        data = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
    else:
        data = yf.download(list(tickers), start=start, end=end)
        data = data.dropna()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            data.to_csv(cache_path)
    return data


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(n).mean()
    roll_down = pd.Series(down, index=series.index).rolling(n).mean()
    rs = (roll_up + 1e-12) / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-12)


def make_features(
    tickers: Sequence[str],
    start: str = "2010-01-01",
    end: str = "2019-01-01",
    feature_list: Iterable[str] = (
        "ret_log",
        "vol_20",
        "hl_range",
        "mom_21",
        "ma_ratio_20",
        "ma_ratio_50",
        "rsi_14",
        "atr_14",
    ),
    normalize: bool = True,
    bench: str = "SPY",
    cache_path: Optional[str] = None,
    force_download: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build the (T, N, F) feature tensor used by the environment.

    The first feature is always the daily log-return of the close price. The
    same feature template is applied to every ticker.
    """

    raw = load_price_data(tickers, start=start, end=end, cache_path=cache_path, force_download=force_download)

    close = raw["Close"]
    open_ = raw["Open"] if "Open" in raw else close
    high = raw["High"] if "High" in raw else close
    low = raw["Low"] if "Low" in raw else close
    volume = raw["Volume"] if "Volume" in raw else None

    bench_close = close[bench] if bench in close.columns else None
    bench_ret = None
    if bench_close is not None:
        bench_ret = np.log(bench_close / bench_close.shift(1))

    per_ticker_frames = []

    for ticker in tickers:
        c = close[ticker]
        o = open_[ticker]
        h = high[ticker]
        l = low[ticker]
        v = volume[ticker] if volume is not None else None

        ret_log = np.log(c / c.shift(1))

        feats = []
        names = []

        def add(name: str, series: pd.Series):
            feats.append(series)
            names.append(f"{ticker}_{name}")

        feats.append(ret_log)
        names.append(f"{ticker}_ret")

        if "vol_20" in feature_list:
            add("vol20", ret_log.rolling(20).std())
        if "hl_range" in feature_list:
            add("hl", (h - l) / (c + 1e-12))
        if "mom_21" in feature_list:
            add("mom21", c.shift(0) / (c.shift(21) + 1e-12) - 1)
        if "mom_63" in feature_list:
            add("mom63", c.shift(0) / (c.shift(63) + 1e-12) - 1)
        if "mom_252" in feature_list:
            add("mom252", c.shift(0) / (c.shift(252) + 1e-12) - 1)
        if "ma_ratio_20" in feature_list:
            add("ma20r", c / (c.rolling(20).mean() + 1e-12) - 1)
        if "ma_ratio_50" in feature_list:
            add("ma50r", c / (c.rolling(50).mean() + 1e-12) - 1)
        if "ma_ratio_200" in feature_list:
            add("ma200r", c / (c.rolling(200).mean() + 1e-12) - 1)
        if "rsi_14" in feature_list:
            add("rsi14", _rsi(c, 14))
        if "atr_14" in feature_list:
            add("atr14", _atr(raw.xs(ticker, level=1, axis=1), 14) / (c + 1e-12))

        if bench_ret is not None:
            if "beta_63" in feature_list:
                cov = ret_log.rolling(63).cov(bench_ret)
                var = bench_ret.rolling(63).var()
                add("beta63", cov / (var + 1e-12))
            if "corr_63" in feature_list:
                add("corr63", ret_log.rolling(63).corr(bench_ret))

        if "overnight_ret" in feature_list:
            add("overnight", np.log(o / (c.shift(1) + 1e-12)))
        if "gap" in feature_list:
            add("gap", (o - c.shift(1)) / (c.shift(1) + 1e-12))

        df_t = pd.concat(feats, axis=1)
        df_t.columns = names

        if normalize:
            for col in df_t.columns:
                if col.endswith("_ret"):
                    continue
                df_t[col] = _rolling_zscore(df_t[col], 60)

        per_ticker_frames.append(df_t)

    df_all = pd.concat(per_ticker_frames, axis=1).dropna()
    n_assets = len(tickers)
    n_features = len(per_ticker_frames[0].columns)
    arr = df_all.values.reshape(len(df_all), n_assets, n_features).astype(np.float32)
    return arr, df_all
