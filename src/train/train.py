import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.utils.download_data import make_features_micro, make_features_macro
from src.envs.portfolio_env import PortfolioEnv
from src.agents.gru_extractor import GRUFeatureExtractor

import yfinance as yf
import pandas as pd
import numpy as np

# --- dans train.py ---
import os
import numpy as np
import pandas as pd
import yfinance as yf




def run_train(csv_returns_path, total_timesteps=100_000, window=20, n_envs=4, out_path="experiments/ppo_gru",i=0):


    tickers = [
    # Tech
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "ADBE", "INTC", "ORCL", "CRM", "AMD",
    "SPY"
    ]
    macro_tickers = ["^VIX", "DX-Y.NYB", "^TNX", "GC=F", "CL=F", "SPY"]

    features_micro, df_micro = make_features_micro(tickers, start="2010-01-01", end="2018-12-31")
    features_eval_micro, df_eval_micro = make_features_micro(tickers, start="2019-01-01", end="2022-05-31")

    features_macro, df_macro = make_features_macro(macro_tickers, start="2010-01-01", end="2018-12-31")
    features_eval_macro, df_eval_macro = make_features_macro(macro_tickers, start="2019-01-01", end="2022-05-31")

    common_index_train = df_micro.index.intersection(df_macro.index)
    df_micro = df_micro.loc[common_index_train]
    df_macro = df_macro.loc[common_index_train]

    common_index_eval = df_eval_micro.index.intersection(df_eval_macro.index)
    df_eval_micro = df_eval_micro.loc[common_index_eval]
    df_eval_macro = df_eval_macro.loc[common_index_eval]
    features_micro = df_micro.values.reshape(len(df_micro), len(tickers), -1).astype(np.float32)
    features_macro = df_macro.values.astype(np.float32)

    features_eval_micro = df_eval_micro.values.reshape(len(df_eval_micro), len(tickers), -1).astype(np.float32)
    features_eval_macro = df_eval_macro.values.astype(np.float32)

    

    def make_env(i_env):
        def _init():
            env = PortfolioEnv(features_micro,features_macro, window=window)
            env.seed(i * 1000 + i_env)                    # <---
            env.action_space.seed(i * 1000 + i_env)       # <---
            return env
        return _init

    env = DummyVecEnv([make_env(j) for j in range(n_envs)])

    policy_kwargs = dict(
        features_extractor_class=GRUFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128, hidden_size=64),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        optimizer_kwargs=dict(weight_decay=1e-4)
    )

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs,seed=i, verbose=1, tensorboard_log="./logs/ppo_gru")

    checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path="./checkpoints/", name_prefix="ppo_gru")
    
    eval_env = PortfolioEnv(features_eval_micro,features_eval_macro, window=window)
    eval_env.seed(10_000 + i)
    eval_env.action_space.seed(10_000 + i)
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./best_model_{i}/",
                                 log_path="./logs/eval_logs/", eval_freq=5_000,
                                 deterministic=True, render=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    model.save(out_path)
    print(f"✅ Training complete. Model saved to {out_path}")
