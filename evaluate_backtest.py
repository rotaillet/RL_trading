# src/train/evaluate_backtest.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.envs.portfolio_env import PortfolioEnv
from src.utils.metrics import sharpe_ratio, max_drawdown, annualized_return
from src.train.train import make_features_macro, make_features_micro

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
        "SPY"
        ]   
    macro_tickers = ["^VIX", "DX-Y.NYB", "^TNX", "GC=F", "CL=F", "SPY"]

    features_micro, df_micro = make_features_micro(tickers, start=start_eval, end=end_eval)


    features_macro, df_macro = make_features_macro(macro_tickers, start=start_eval, end=end_eval)

    common_index_train = df_micro.index.intersection(df_macro.index)
    df_micro = df_micro.loc[common_index_train]
    df_macro = df_macro.loc[common_index_train]

    features_micro = df_micro.values.reshape(len(df_micro), len(tickers), -1).astype(np.float32)
    features_macro = df_macro.values.astype(np.float32)


    # === Features train/eval (eval pour l'env; train pas indispensable ici, mais utile si tu veux aligner les périodes)
    # On utilise surtout features_eval (T_eval, N, F) ; F[0] = log-return par construction de make_features

    env = PortfolioEnv(features_micro,features_macro, window=window)

    # === Chargement du modèle
    model = PPO.load(model_path)
    k = 5
    # === Simulation RL
    obs = env.reset()
    done = False
    daily_logrets_rl = []     # log-return net (du jour) reporté par l'env (via net_daily_logret dans step)
    values_rl = []            # valeur du portefeuille (arithmétique)
    turnovers = []
    weights_list = []  # <-- new

    hold_counter = 0
    hold_action = None

    while not done:
        if hold_counter == 0:
        # nouvelle action
            hold_action, _ = model.predict(obs, deterministic=True)
            hold_counter = k  # on va garder cette action pendant k steps
        action = hold_action
        hold_counter -= 1
        obs, reward, done, info = env.step(action)
        # l'env met à jour la valeur en arithmétique -> on la lit dans info
        values_rl.append(info.get("portfolio_value", env.value))
        # on reconstitue le log-return net du jour à partir de la valeur (safe si besoin)
        if len(values_rl) >= 2:
            v_prev, v_now = values_rl[-2], values_rl[-1]
            daily_logrets_rl.append(np.log((v_now + 1e-12) / (v_prev + 1e-12)))
        turnovers.append(info.get("turnover", 0.0))
        weights_list.append(info.get("weights")) 
        print(weights_list[-1])
    values_rl = np.array(values_rl, dtype=float) 
    weights_arr = np.array(weights_list, dtype=float)  # shape: (T, N)

    print(weights_arr.shape)
    
    plt.figure(figsize=(10, 5))
    for i, t in enumerate(tickers):
        plt.plot(weights_arr[:, i], label=t, alpha=0.7)
    plt.legend(ncol=2)
    plt.title("Poids du portefeuille par actif (RL)")
    plt.xlabel("Temps (pas d'episode)")
    plt.ylabel("Poids")
    plt.tight_layout()
    plt.savefig("images/allocation_par_actif.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.stackplot(range(weights_arr.shape[0]), weights_arr.T, labels=tickers)
    plt.legend(ncol=2, loc="upper left")
    plt.title("Allocation cumulée (stacked) du portefeuille RL")
    plt.xlabel("Temps")
    plt.ylabel("Poids cumulés")
    plt.tight_layout()
    plt.savefig("images/allocation_stack.png", dpi=120)
    plt.close()

    turnovers = np.array(turnovers, dtype=float)

    # Série temporelle
    plt.figure(figsize=(10, 4))
    plt.plot(turnovers)
    plt.title("Turnover quotidien (L1) du portefeuille RL")
    plt.xlabel("Temps")
    plt.ylabel("Turnover L1")
    plt.tight_layout()
    plt.savefig("images/turnover_series.png", dpi=120)
    plt.close()

    # Histogramme
    plt.figure(figsize=(6, 4))
    plt.hist(turnovers, bins=30)
    plt.title("Distribution du turnover quotidien (RL)")
    plt.xlabel("Turnover L1")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig("images/turnover_hist.png", dpi=120)
    plt.close()


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
    ret_cols = [c for c in df_micro.columns if c.endswith("_ret")]
    # Mapping pour retrouver le ticker depuis "TICKER_ret"
    col_map = {c.split("_ret")[0]: c for c in ret_cols}

    # Equal-Weight baseline
    tickers_for_ew = tickers[:] if include_spy_in_ew else [t for t in tickers if t != "SPY"]
    ret_logs_mat = []
    for t in tickers_for_ew:
        if t in col_map:
            ret_logs_mat.append(df_micro[col_map[t]].values)
    ret_logs_mat = np.vstack(ret_logs_mat)  # shape: (N_ew, T_eval)
    # moyenne des log-returns -> approx log-return du panier égal-pondéré
    ew_log_full = ret_logs_mat.mean(axis=0)
    ew_log = ew_log_full[window:]  # on aligne avec RL (démarre à t=window-1)
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
        spy_log_full = df_micro[col_map["SPY"]].values
        spy_log = spy_log_full[window:]  # aligne avec RL
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

    plt.figure(figsize=(10, 4))
    plt.hist(rl_daily_arith, bins=50, alpha=0.5, label="RL")
    plt.hist(ew_arith, bins=50, alpha=0.5, label="EW")
    if spy_curve is not None:
        plt.hist(spy_arith, bins=50, alpha=0.5, label="SPY")
    plt.legend()
    plt.title("Distribution des rendements journaliers")
    plt.xlabel("Rendement journalier")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig("images/returns_hist.png", dpi=120)
    plt.close()

    if spy_curve is not None:
        df_ret = pd.DataFrame({
            "RL": rl_daily_arith,
            "EW": ew_arith,
            "SPY": spy_arith
        })

        corr_with_spy = df_ret.corr()["SPY"]
        print("\nCorrélations avec SPY:")
        print(corr_with_spy)

        # Scatter RL vs SPY
        plt.figure(figsize=(5,5))
        plt.scatter(df_ret["SPY"], df_ret["RL"], s=5, alpha=0.5)
        plt.xlabel("SPY return")
        plt.ylabel("RL return")
        plt.title("RL vs SPY (corrélation / bêta visuel)")
        plt.tight_layout()
        plt.savefig("images/rl_vs_spy_scatter.png", dpi=120)
        plt.close()
    plt.figure(figsize=(10, 6))
    plt.imshow(weights_arr.T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Poids")
    plt.yticks(range(len(tickers)), tickers)
    plt.xlabel("Temps")
    plt.title("Heatmap des poids du portefeuille (RL)")
    plt.tight_layout()
    plt.savefig("images/weights_heatmap.png", dpi=120)
    plt.close()



    return results, dict(rl_curve=values_rl, ew_curve=ew_curve, spy_curve=spy_curve)

from stable_baselines3 import PPO
from src.envs.portfolio_env import PortfolioEnv
from src.utils.metrics import sharpe_ratio, max_drawdown, annualized_return

def simulate_rl_on_features(model_path, features_eval, window=60):
    env = PortfolioEnv(features_eval, window=window)
    model = PPO.load(model_path)

    obs = env.reset()
    done = False
    values_rl = []
    daily_logrets_rl = []
    turnovers = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        values_rl.append(info.get("portfolio_value", env.value))
        if len(values_rl) >= 2:
            v_prev, v_now = values_rl[-2], values_rl[-1]
            daily_logrets_rl.append(np.log((v_now + 1e-12) / (v_prev + 1e-12)))
        turnovers.append(info.get("turnover", 0.0))

    values_rl = np.array(values_rl, dtype=float)
    rl_daily_arith = np.exp(np.array(daily_logrets_rl, dtype=float)) - 1.0

    metrics = {
        "Final value": values_rl[-1],
        "CAGR": annualized_return(values_rl),
        "Sharpe": sharpe_ratio(rl_daily_arith),
        "MaxDD": max_drawdown(values_rl),
        "Turnover (avg L1)": float(np.mean(turnovers)) if len(turnovers) > 0 else 0.0,
    }
    return metrics, np.array(daily_logrets_rl, dtype=float)

def randomization_test_perm_tickers(model_path, features_eval, window=60, n_runs=30):
    base_metrics, base_logrets = simulate_rl_on_features(model_path, features_eval, window)
    print("Baseline Sharpe:", base_metrics["Sharpe"], "CAGR:", base_metrics["CAGR"])

    sharpe_null = []
    cagr_null = []

    T, N, F = features_eval.shape

    for k in range(n_runs):
        perm = np.random.permutation(N)
        feats_rand = features_eval[:, perm, :].copy()
        m, _ = simulate_rl_on_features(model_path, feats_rand, window)
        sharpe_null.append(m["Sharpe"])
        cagr_null.append(m["CAGR"])

    sharpe_null = np.array(sharpe_null)
    cagr_null = np.array(cagr_null)

    print("\nRandomization (perm_tickers) —", n_runs, "runs")
    print("Sharpe null mean / std:", sharpe_null.mean(), sharpe_null.std())
    print("CAGR   null mean / std:", cagr_null.mean(), cagr_null.std())

    return base_metrics, sharpe_null, cagr_null
def randomization_test_time_roll(model_path, features_eval, window=60, n_runs=30):
    base_metrics, base_logrets = simulate_rl_on_features(model_path, features_eval, window)
    print("Baseline Sharpe:", base_metrics["Sharpe"], "CAGR:", base_metrics["CAGR"])

    sharpe_null = []
    cagr_null = []

    T = features_eval.shape[0]
    for k in range(n_runs):
        shift = np.random.randint(window, T - window)
        feats_rand = np.roll(features_eval, shift=shift, axis=0)
        m, _ = simulate_rl_on_features(model_path, feats_rand, window)
        sharpe_null.append(m["Sharpe"])
        cagr_null.append(m["CAGR"])

    sharpe_null = np.array(sharpe_null)
    cagr_null = np.array(cagr_null)

    print("\nRandomization (time_roll) —", n_runs, "runs")
    print("Sharpe null mean / std:", sharpe_null.mean(), sharpe_null.std())
    print("CAGR   null mean / std:", cagr_null.mean(), cagr_null.std())

    return base_metrics, sharpe_null, cagr_null
from src.utils.metrics import sharpe_ratio, max_drawdown

def _cumcurve_from_arith_returns(r):
    r = np.asarray(r, dtype=float)
    return np.cumprod(1.0 + r)

def bootstrap_performance(daily_logrets, n_boot=1000):
    daily_arith = np.exp(daily_logrets) - 1.0
    T = len(daily_arith)

    finals = []
    sharpes = []
    mdds = []

    for b in range(n_boot):
        idx = np.random.randint(0, T, size=T)  # sampling with replacement
        r_b = daily_arith[idx]
        curve_b = _cumcurve_from_arith_returns(r_b)
        finals.append(curve_b[-1])
        sharpes.append(sharpe_ratio(r_b))
        mdds.append(max_drawdown(curve_b))

    return np.array(finals), np.array(sharpes), np.array(mdds)
def analyze_bootstrap(model_path, features_eval, window=60, n_boot=1000):
    base_metrics, base_logrets = simulate_rl_on_features(model_path, features_eval, window)

    finals_b, sharpes_b, mdds_b = bootstrap_performance(base_logrets, n_boot=n_boot)

    print("=== Baseline (réel) ===")
    print(base_metrics)

    print("\n=== Bootstrap ===")
    print("Final value 95th pct:", np.percentile(finals_b, 95))
    print("Sharpe      95th pct:", np.percentile(sharpes_b, 95))
    print("MaxDD       5th pct :", np.percentile(mdds_b, 5))

    # “p-value” empirique rudimentaire
    p_sharpe = (sharpes_b >= base_metrics["Sharpe"]).mean()
    p_final = (finals_b >= base_metrics["Final value"]).mean()

    print("\nEmpirical p-values:")
    print("p(Sharpe_boot >= Sharpe_real) =", p_sharpe)
    print("p(Final_boot >= Final_real)   =", p_final)

    return {
        "base_metrics": base_metrics,
        "finals_boot": finals_b,
        "sharpes_boot": sharpes_b,
        "mdds_boot": mdds_b,
        "p_sharpe": p_sharpe,
        "p_final": p_final,
    }


if __name__ == "__main__":
    # Exemple d’utilisation :
    # 1) modèle "best_model/best_model.zip"
    # 2) autre checkpoint
    print("Running evaluate_backtest ...")
    tickers = [
        "AAPL","MSFT","GOOG","AMZN","NVDA","META","ADBE","INTC","ORCL","CRM","AMD",
        "SPY"
    ]

    #features_eval, df_eval = make_features(tickers, start="2019-01-01", end="2025-11-01")

    model_path = "best_model/best_model.zip"

    # 1) Test de randomisation (permute tickers)
    #randomization_test_perm_tickers(model_path, features_eval, window=60, n_runs=30)

    # 2) Test de randomisation (time roll)
    #randomization_test_time_roll(model_path, features_eval, window=60, n_runs=30)

    # 3) Bootstrapping sur la trajectoire réelle
    #analyze_bootstrap(model_path, features_eval, window=60, n_boot=1000)
    # backtest best model
    backtest(
        model_path="best_model_0/best_model.zip",
        tickers=tickers,
        start_eval="2022-06-01",
        end_eval="2025-11-01",
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
