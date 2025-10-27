import os
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.agents.transformer_extractor import TransformerFeatureExtractor
from src.envs.portfolio_env import PortfolioEnv
from src.pretraining.encoder_pretrain import PretrainConfig, pretrain_encoder
from src.train.train import run_train
from src.utils.data import make_features

DEFAULT_RL_TICKERS = (
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "NVDA",
    "META",
    "ADBE",
    "INTC",
    "ORCL",
    "CRM",
    "AMD",
    "SPY",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "TSLA",
    "HD",
    "PG",
    "KO",
)

DEFAULT_PRETRAIN_TICKERS = tuple(
    dict.fromkeys(
        list(DEFAULT_RL_TICKERS)
        + [
            "UNH",
            "PFE",
            "JNJ",
            "MRK",
            "ABBV",
            "GS",
            "MS",
            "C",
            "WMT",
            "COST",
            "PEP",
            "NKE",
            "DIS",
            "NFLX",
            "CMCSA",
            "BA",
            "CAT",
            "DE",
            "LMT",
            "HON",
            "SLB",
            "EOG",
            "COP",
            "V",
            "MA",
            "PYPL",
            "ADP",
            "IBM",
            "QCOM",
            "TXN",
            "AMAT",
            "LRCX",
            "MU",
            "BKNG",
            "SBUX",
            "MCD",
            "GE",
            "NOW",
            "SHOP",
            "UBER",
        ]
    )
)


@dataclass
class PipelineConfig:
    pretrain_tickers: Sequence[str]
    rl_tickers: Sequence[str]
    data_dir: str = "data"
    window: int = 20
    pretrain_train_period: Tuple[str, str] = ("2004-01-01", "2018-12-31")
    pretrain_val_period: Tuple[str, str] = ("2019-01-01", "2020-12-31")
    rl_train_period: Tuple[str, str] = ("2010-01-01", "2018-12-31")
    rl_val_period: Tuple[str, str] = ("2019-01-01", "2021-12-31")
    rl_test_period: Tuple[str, str] = ("2022-01-01", "2024-12-31")
    encoder_checkpoint: str = "checkpoints/pretrained_encoder.pt"
    policy_checkpoint: str = "experiments/ppo_transformer"
    total_timesteps: int = 200_000
    n_envs: int = 4
    pretrain_epochs: int = 25
    pretrain_batch_size: int = 64
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 1e-4
    transformer_d_model: int = 128
    transformer_time_layers: int = 2
    transformer_asset_layers: int = 1
    transformer_heads: int = 4
    transformer_dropout: float = 0.1
    freeze_encoder: bool = False

    def __post_init__(self):
        self.pretrain_tickers = tuple(dict.fromkeys(self.pretrain_tickers))
        self.rl_tickers = tuple(dict.fromkeys(self.rl_tickers))


def _cache_path(data_dir: str, label: str, start: str, end: str) -> str:
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{label}_{start}_{end}.csv")


def _build_features(
    tickers: Sequence[str],
    start: str,
    end: str,
    data_dir: str,
    label: str,
    feature_list: Iterable[str],
) -> np.ndarray:
    cache_path = _cache_path(data_dir, label, start, end)
    features, _ = make_features(
        tickers,
        start=start,
        end=end,
        feature_list=feature_list,
        cache_path=cache_path,
    )
    return features


def _evaluate_model(model: PPO, features: np.ndarray, window: int) -> Dict[str, float]:
    env = PortfolioEnv(features, window=window)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
    values = env.get_value_series()
    returns = np.diff(np.log(values + 1e-12))
    total_return = values[-1] / values[0] - 1
    cagr = (values[-1] / values[0]) ** (252 / max(len(returns), 1)) - 1 if len(returns) > 0 else 0.0
    sharpe = (np.mean(returns) / (np.std(returns) + 1e-12)) * np.sqrt(252) if len(returns) > 1 else 0.0
    max_dd = float(env.max_drawdown)
    return {
        "final_value": float(values[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }


def run_full_pipeline(config: PipelineConfig) -> Dict[str, Dict[str, float]]:
    os.makedirs(config.data_dir, exist_ok=True)

    pretrain_cfg = PretrainConfig(
        tickers=config.pretrain_tickers,
        train_period=config.pretrain_train_period,
        val_period=config.pretrain_val_period,
        window=config.window,
        epochs=config.pretrain_epochs,
        batch_size=config.pretrain_batch_size,
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
        d_model=config.transformer_d_model,
        time_layers=config.transformer_time_layers,
        asset_layers=config.transformer_asset_layers,
        time_heads=config.transformer_heads,
        asset_heads=config.transformer_heads,
        dropout=config.transformer_dropout,
        max_assets=max(len(config.pretrain_tickers), len(config.rl_tickers)),
        save_path=config.encoder_checkpoint,
    )

    pretrain_train_features = _build_features(
        config.pretrain_tickers,
        config.pretrain_train_period[0],
        config.pretrain_train_period[1],
        config.data_dir,
        "pretrain_train",
        pretrain_cfg.feature_list,
    )
    pretrain_val_features = _build_features(
        config.pretrain_tickers,
        config.pretrain_val_period[0],
        config.pretrain_val_period[1],
        config.data_dir,
        "pretrain_val",
        pretrain_cfg.feature_list,
    )

    encoder_path = pretrain_encoder(pretrain_cfg, pretrain_train_features, pretrain_val_features)

    # --- RL fine-tuning features ---
    rl_train_features = _build_features(
        config.rl_tickers,
        config.rl_train_period[0],
        config.rl_train_period[1],
        config.data_dir,
        "rl_train",
        pretrain_cfg.feature_list,
    )
    rl_val_features = _build_features(
        config.rl_tickers,
        config.rl_val_period[0],
        config.rl_val_period[1],
        config.data_dir,
        "rl_val",
        pretrain_cfg.feature_list,
    )
    rl_test_features = _build_features(
        config.rl_tickers,
        config.rl_test_period[0],
        config.rl_test_period[1],
        config.data_dir,
        "rl_test",
        pretrain_cfg.feature_list,
    )

    features_kwargs = dict(
        features_dim=256,
        d_model=config.transformer_d_model,
        time_layers=config.transformer_time_layers,
        asset_layers=config.transformer_asset_layers,
        time_heads=config.transformer_heads,
        asset_heads=config.transformer_heads,
        dropout=config.transformer_dropout,
        max_assets=max(len(config.pretrain_tickers), len(config.rl_tickers)),
        pretrained_path=encoder_path,
        freeze_encoder=config.freeze_encoder,
    )

    model = run_train(
        train_features=rl_train_features,
        eval_features=rl_val_features,
        total_timesteps=config.total_timesteps,
        window=config.window,
        n_envs=config.n_envs,
        out_path=config.policy_checkpoint,
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=features_kwargs,
    )

    metrics = {
        "validation": _evaluate_model(model, rl_val_features, config.window),
        "test": _evaluate_model(model, rl_test_features, config.window),
    }

    return metrics
