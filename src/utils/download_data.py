import yfinance as yf
import pandas as pd
import numpy as np
import os

def _rsi(series, n=14):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(n).mean()
    roll_down = pd.Series(down, index=series.index).rolling(n).mean()
    rs = (roll_up + 1e-12) / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _atr(df, n=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def make_features_micro(
    tickers,
    start="2010-01-01",
    end="2019-01-01",
    feature_list=(
        "ret_log",        # gardé en position 0
        "vol_20",
        "hl_range",
        "mom_21",
        "ma_ratio_20",
        "ma_ratio_50",
        "rsi_14",
        "atr_14",
        # "beta_63",      # nécessite bench présent
        # "corr_63",      # nécessite bench présent
        # "overnight_ret",
        # "gap",
    ),
    normalize=True,
    bench="SPY",
    cache_path=None
):
    # --- data loading + cache ---
    if cache_path and os.path.exists(cache_path):
        data = pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)
    else:
        data = yf.download(tickers, start=start, end=end)
        data = data
        if cache_path:
            data.to_csv(cache_path)

    # multiindex columns: ('Adj Close', 'AAPL'), ...
    close = data["Close"]
    open_ = data["Open"] if "Open" in data else close
    high = data["High"] if "High" in data else close
    low  = data["Low"]  if "Low"  in data else close
    volu = data["Volume"] if "Volume" in data else None

    # benchmark series if requested
    bench_close = close[bench] if (bench in close.columns) else None
    bench_ret = None
    if bench_close is not None:
        bench_ret = np.log(bench_close/bench_close.shift(1))

    per_ticker_frames = []

    for t in tickers:
        c = close[t]
        o = open_[t]
        h = high[t]
        l = low[t]
        v = volu[t] if volu is not None else None

        # base returns (log)
        ret_log = np.log(c/c.shift(1))

        feats = []
        names = []

        # 0) ret_log ALWAYS first
        feats.append(ret_log)
        names.append(f"{t}_ret")

        # --- helpers ---
        def add(name, series):
            feats.append(series)
            names.append(f"{t}_{name}")

        # 1) volatility & range
        if "vol_20" in feature_list:
            add("vol20", ret_log.rolling(20).std())
        if "hl_range" in feature_list:
            add("hl", (h - l) / c)

        # 2) momentum / trend
        if "mom_21" in feature_list:
            add("mom21", c.shift(0)/c.shift(21) - 1)
        if "mom_63" in feature_list:
            add("mom63", c.shift(0)/c.shift(63) - 1)
        if "mom_252" in feature_list:
            add("mom252", c.shift(0)/c.shift(252) - 1)
        if "ma_ratio_20" in feature_list:
            add("ma20r", c / (c.rolling(20).mean() + 1e-12) - 1)
        if "ma_ratio_50" in feature_list:
            add("ma50r", c / (c.rolling(50).mean() + 1e-12) - 1)
        if "ma_ratio_200" in feature_list:
            add("ma200r", c / (c.rolling(200).mean() + 1e-12) - 1)
        if "rsi_14" in feature_list:
            add("rsi14", _rsi(c, 14))

        # 3) volatility robust
        if "atr_14" in feature_list:
            add("atr14", _atr(data.xs(t, level=1, axis=1), 14) / (c + 1e-12))

        # 4) risk to benchmark
        if bench_ret is not None:
            if "beta_63" in feature_list:
                # beta = cov(ret, ret_bench)/var(ret_bench)
                cov = ret_log.rolling(63).cov(bench_ret)
                var = bench_ret.rolling(63).var()
                add("beta63", cov / (var + 1e-12))
            if "corr_63" in feature_list:
                add("corr63", ret_log.rolling(63).corr(bench_ret))

        # 5) micro (open/close)
        if "overnight_ret" in feature_list:
            add("overnight", np.log(o / c.shift(1)))
        if "gap" in feature_list:
            add("gap", (o - c.shift(1)) / (c.shift(1) + 1e-12))

        df_t = pd.concat(feats, axis=1)
        df_t.columns = names

        # --- normalisation (optionnelle) ---
        if normalize:
            # ret_log : z-score rolling 60j
            # pour toutes les colonnes sauf *_ret (déjà z-score), on applique une normalisation "safe"
            for col in df_t.columns:
                if col.endswith("_ret"):
                    continue
                s = df_t[col]
                # heuristique: si déjà ratio ou corr (-1..+1), z-score léger ; sinon z-score standard
                df_t[col] = (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-12)

        per_ticker_frames.append(df_t)

    # concat par colonnes puis reshape (T, N, F)
    df_all = pd.concat(per_ticker_frames, axis=1)
    # IMPORTANT: on récupère l'ordre des features par actif pour construire F
    # (on suppose que chaque actif a le même set)
    N = len(tickers)
    F = len(per_ticker_frames[0].columns)  # ret en 0
    arr = df_all.values.reshape(len(df_all), N, F).astype(np.float32)
    return arr, df_all

def make_features_macro(
    tickers,
    start="2010-01-01",
    end="2019-01-01",
    feature_list=(
        "ret_log",        # gardé en position 0
        "vol_20",
        "hl_range",
        "mom_21",
        "ma_ratio_20",
        "ma_ratio_50",
        "rsi_14",
        "atr_14",
        # "beta_63",      # nécessite bench présent
        # "corr_63",      # nécessite bench présent
        # "overnight_ret",
        # "gap",
    ),
    normalize=True,
    bench="SPY",
    cache_path=None
):
    # --- data loading + cache ---
    if cache_path and os.path.exists(cache_path):
        data = pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)
    else:
        data = yf.download(tickers, start=start, end=end)
        data = data.dropna()
        if cache_path:
            data.to_csv(cache_path)

    # multiindex columns: ('Adj Close', 'AAPL'), ...
    close = data["Close"]
    open_ = data["Open"] if "Open" in data else close
    high = data["High"] if "High" in data else close
    low  = data["Low"]  if "Low"  in data else close
    volu = data["Volume"] if "Volume" in data else None

    # benchmark series if requested
    bench_close = close[bench] if (bench in close.columns) else None
    bench_ret = None
    if bench_close is not None:
        bench_ret = np.log(bench_close/bench_close.shift(1))

    per_ticker_frames = []

    for t in tickers:
        c = close[t]
        o = open_[t]
        h = high[t]
        l = low[t]
        v = volu[t] if volu is not None else None

        # base returns (log)
        ret_log = np.log(c/c.shift(1))

        feats = []
        names = []

        # 0) ret_log ALWAYS first
        feats.append(ret_log)
        names.append(f"{t}_ret")

        # --- helpers ---
        def add(name, series):
            feats.append(series)
            names.append(f"{t}_{name}")

        # 1) volatility & range
        if "vol_20" in feature_list:
            add("vol20", ret_log.rolling(20).std())
        if "hl_range" in feature_list:
            add("hl", (h - l) / c)

        # 2) momentum / trend
        if "mom_21" in feature_list:
            add("mom21", c.shift(0)/c.shift(21) - 1)
        if "mom_63" in feature_list:
            add("mom63", c.shift(0)/c.shift(63) - 1)
        if "mom_252" in feature_list:
            add("mom252", c.shift(0)/c.shift(252) - 1)
        if "ma_ratio_20" in feature_list:
            add("ma20r", c / (c.rolling(20).mean() + 1e-12) - 1)
        if "ma_ratio_50" in feature_list:
            add("ma50r", c / (c.rolling(50).mean() + 1e-12) - 1)
        if "ma_ratio_200" in feature_list:
            add("ma200r", c / (c.rolling(200).mean() + 1e-12) - 1)
        if "rsi_14" in feature_list:
            add("rsi14", _rsi(c, 14))

        # 3) volatility robust
        if "atr_14" in feature_list:
            add("atr14", _atr(data.xs(t, level=1, axis=1), 14) / (c + 1e-12))

        # 4) risk to benchmark
        if bench_ret is not None:
            if "beta_63" in feature_list:
                # beta = cov(ret, ret_bench)/var(ret_bench)
                cov = ret_log.rolling(63).cov(bench_ret)
                var = bench_ret.rolling(63).var()
                add("beta63", cov / (var + 1e-12))
            if "corr_63" in feature_list:
                add("corr63", ret_log.rolling(63).corr(bench_ret))

        # 5) micro (open/close)
        if "overnight_ret" in feature_list:
            add("overnight", np.log(o / c.shift(1)))
        if "gap" in feature_list:
            add("gap", (o - c.shift(1)) / (c.shift(1) + 1e-12))

        df_t = pd.concat(feats, axis=1)
        df_t.columns = names

        # --- normalisation (optionnelle) ---
        if normalize:
            # ret_log : z-score rolling 60j
            # pour toutes les colonnes sauf *_ret (déjà z-score), on applique une normalisation "safe"
            for col in df_t.columns:
                if col.endswith("_ret"):
                    continue
                s = df_t[col]
                # heuristique: si déjà ratio ou corr (-1..+1), z-score léger ; sinon z-score standard
                df_t[col] = (s - s.rolling(60).mean()) / (s.rolling(60).std() + 1e-12)

        per_ticker_frames.append(df_t)

    # concat par colonnes puis reshape (T, N, F)
    df_all = pd.concat(per_ticker_frames, axis=1).dropna()
    # IMPORTANT: on récupère l'ordre des features par actif pour construire F
    # (on suppose que chaque actif a le même set)
    N = len(tickers)
    F = len(per_ticker_frames[0].columns)  # ret en 0
    arr = df_all.values.astype(np.float32)
    return arr, df_all


