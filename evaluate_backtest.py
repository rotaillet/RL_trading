# src/train/evaluate_backtest.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.envs.portfolio_env import PortfolioEnv
from src.utils.metrics import sharpe_ratio, max_drawdown, annualized_return
from src.train.train import make_features

def _cumcurve_from_arith_returns(r):
    """r: array-like de retours arithmétiques journaliers -> courbe de valeur (start=1)."""
    r = np.asarray(r, dtype=float)
    v = np.cumprod(1.0 + r)
    return v

def backtest(
    model_path: str,
    tickers=None,
    start_train="2010-01-01",
    end_train="2018-12-31",
    start_eval="2019-01-01",
    end_eval="2024-12-31",
    window=60,
    include_spy_in_ew=True,
    image_name="backtest_eval"
):
    """
    Charge le modèle PPO, lance un backtest sur la période d'éval,
    et compare à Equal-Weight et SPY (buy&hold).
    - Les retours de l'agent et des baselines sont mesurés en ARITHMÉTIQUE pour Sharpe.
    """
    if tickers is None:
        # Liste courte par défaut ; n'oublie pas de garder 'SPY' si tu veux beta/corr ou baseline SPY
        tickers = [
            # Tech
            "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "ADBE", "INTC", "ORCL", "CRM", "AMD",
            # Benchmark
            "SPY"
        ]

    # === Features train/eval (eval pour l'env; train pas indispensable ici, mais utile si tu veux aligner les périodes)
    # On utilise surtout features_eval (T_eval, N, F) ; F[0] = log-return par construction de make_features
    features_eval, df_eval = make_features(tickers, start=start_eval, end=end_eval)
    env = PortfolioEnv(features_eval, window=window)

    # === Chargement du modèle
    model = PPO.load(model_path)

    # === Simulation RL
    obs = env.reset()
    done = False
    daily_logrets_rl = []     # log-return net (du jour) reporté par l'env (via net_daily_logret dans step)
    values_rl = []            # valeur du portefeuille (arithmétique)
    turnovers = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # l'env met à jour la valeur en arithmétique -> on la lit dans info
        values_rl.append(info.get("portfolio_value", env.value))
        # on reconstitue le log-return net du jour à partir de la valeur (safe si besoin)
        if len(values_rl) >= 2:
            v_prev, v_now = values_rl[-2], values_rl[-1]
            daily_logrets_rl.append(np.log((v_now + 1e-12) / (v_prev + 1e-12)))
        turnovers.append(info.get("turnover", 0.0))

    values_rl = np.array(values_rl, dtype=float)
    # Convertit les log-returns du RL en arithmétique pour la métrique Sharpe
    rl_daily_arith = np.exp(np.array(daily_logrets_rl, dtype=float)) - 1.0

    rl_metrics = {
        "Final value": values_rl[-1],
        "CAGR": annualized_return(values_rl),
        "Sharpe": sharpe_ratio(rl_daily_arith),
        "MaxDD": max_drawdown(values_rl),
        "Turnover (avg L1)": float(np.mean(turnovers)) if len(turnovers) > 0 else 0.0,
    }

    # === Baselines depuis df_eval
    # df_eval contient, pour chaque ticker t, des colonnes t_ret (log), t_vol, t_hl, ...
    ret_cols = [c for c in df_eval.columns if c.endswith("_ret")]
    # Mapping pour retrouver le ticker depuis "TICKER_ret"
    col_map = {c.split("_ret")[0]: c for c in ret_cols}

    # Equal-Weight baseline
    tickers_for_ew = tickers[:] if include_spy_in_ew else [t for t in tickers if t != "SPY"]
    ret_logs_mat = []
    for t in tickers_for_ew:
        if t in col_map:
            ret_logs_mat.append(df_eval[col_map[t]].values)
    ret_logs_mat = np.vstack(ret_logs_mat)  # shape: (N_ew, T_eval)
    # moyenne des log-returns -> approx log-return du panier égal-pondéré
    ew_log = ret_logs_mat.mean(axis=0)
    ew_arith = np.exp(ew_log) - 1.0
    ew_curve = _cumcurve_from_arith_returns(ew_arith)
    ew_metrics = {
        "Final value": float(ew_curve[-1]),
        "CAGR": annualized_return(ew_curve),
        "Sharpe": sharpe_ratio(ew_arith),
        "MaxDD": max_drawdown(ew_curve),
        "Turnover (avg L1)": 0.0,  # buy & hold égal-pondéré sans réallocs
    }

    # SPY buy & hold
    if "SPY" in col_map:
        spy_log = df_eval[col_map["SPY"]].values
        spy_arith = np.exp(spy_log) - 1.0
        spy_curve = _cumcurve_from_arith_returns(spy_arith)
        spy_metrics = {
            "Final value": float(spy_curve[-1]),
            "CAGR": annualized_return(spy_curve),
            "Sharpe": sharpe_ratio(spy_arith),
            "MaxDD": max_drawdown(spy_curve),
            "Turnover (avg L1)": 0.0,
        }
    else:
        spy_curve = None
        spy_metrics = {
            "Final value": np.nan,
            "CAGR": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "Turnover (avg L1)": 0.0,
        }

    # === Résultats
    results = pd.DataFrame(
        [rl_metrics, ew_metrics, spy_metrics],
        index=["RL Portfolio", f"Equal Weight ({'with' if include_spy_in_ew else 'no'} SPY)", "SPY"]
    )
    print("\n=== Résultats Backtest (eval) ===")
    print(results.round(4))

    # === Plot
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(values_rl, label="RL portfolio")
    plt.plot(ew_curve, label=f"Equal weight ({'with' if include_spy_in_ew else 'no'} SPY)")
    if spy_curve is not None:
        plt.plot(spy_curve, label="SPY buy & hold")
    plt.legend()
    plt.title(f"Equity Curves — eval {start_eval} → {end_eval}")
    plt.tight_layout()
    plt.savefig(f"images/{image_name}.png", dpi=120)
    plt.close()

    return results, dict(rl_curve=values_rl, ew_curve=ew_curve, spy_curve=spy_curve)

if __name__ == "__main__":
    # Exemple d’utilisation :
    # 1) modèle "best_model/best_model.zip"
    # 2) autre checkpoint
    print("Running evaluate_backtest ...")
    tickers = [
        "AAPL","MSFT","GOOG","AMZN","NVDA","META","ADBE","INTC","ORCL","CRM","AMD",
        "SPY"
    ]
    # backtest best model
    backtest(
        model_path="best_model/best_model.zip",
        tickers=tickers,
        start_eval="2019-01-01",
        end_eval="2024-12-31",
        window=60,
        include_spy_in_ew=True,
        image_name="eval_bestmodel"
    )
    # backtest d’un autre checkpoint (optionnel)
    # backtest(
    #     model_path="checkpoints/ppo_gru_4800000_steps.zip",
    #     tickers=tickers,
    #     start_eval="2019-01-01",
    #     end_eval="2024-12-31",
    #     window=60,
    #     include_spy_in_ew=False,
    #     image_name="eval_ckpt"
    # )
