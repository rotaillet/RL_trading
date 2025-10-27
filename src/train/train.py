import os
from typing import Optional, Sequence, Tuple, Type

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.gru_extractor import GRUFeatureExtractor
from src.envs.portfolio_env import PortfolioEnv
from src.utils.data import make_features


def _build_env(features: "np.ndarray", window: int) -> PortfolioEnv:
    from src.envs.portfolio_env import PortfolioEnv  # local import to avoid circular deps

    return PortfolioEnv(features, window=window)


def run_train(
    csv_returns_path: Optional[str] = None,
    total_timesteps: int = 100_000,
    window: int = 20,
    n_envs: int = 4,
    out_path: str = "experiments/ppo_gru",
    train_features: Optional["np.ndarray"] = None,
    eval_features: Optional["np.ndarray"] = None,
    features_extractor_class: Type[BaseFeaturesExtractor] = GRUFeatureExtractor,
    features_extractor_kwargs: Optional[dict] = None,
    policy_kwargs_extra: Optional[dict] = None,
    tickers: Optional[Sequence[str]] = None,
    train_period: Tuple[str, str] = ("2010-01-01", "2018-12-31"),
    eval_period: Tuple[str, str] = ("2019-01-01", "2024-12-31"),
) -> PPO:
    """Train a PPO agent on the portfolio environment.

    Parameters
    ----------
    csv_returns_path:
        Deprecated placeholder kept for backward compatibility.
    train_features / eval_features:
        Optional pre-computed feature tensors with shape (T, N, F). If not
        provided they will be generated using `make_features` and the provided
        tickers/periods.
    features_extractor_class / kwargs:
        Custom feature extractor configuration. Defaults to the historical GRU
        extractor.
    policy_kwargs_extra:
        Additional entries merged into the `policy_kwargs` dict passed to PPO.
    tickers:
        Universe used when on-the-fly feature generation is required.
    train_period / eval_period:
        Date ranges used to compute features when `train_features` or
        `eval_features` are not provided.
    """

    del csv_returns_path  # kept for CLI compatibility

    if train_features is None or eval_features is None:
        if not tickers:
            tickers = [
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
            ]

    if train_features is None:
        train_features, _ = make_features(tickers, start=train_period[0], end=train_period[1])
    if eval_features is None:
        eval_features, _ = make_features(tickers, start=eval_period[0], end=eval_period[1])

    def make_env(features_array):
        def _init():
            return _build_env(features_array, window=window)

        return _init

    env = DummyVecEnv([make_env(train_features) for _ in range(n_envs)])

    default_features_kwargs = dict(features_dim=128, hidden_size=64)
    if features_extractor_kwargs:
        default_features_kwargs.update(features_extractor_kwargs)

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=default_features_kwargs,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        optimizer_kwargs=dict(weight_decay=1e-4),
    )

    if policy_kwargs_extra:
        policy_kwargs.update(policy_kwargs_extra)

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/ppo_train",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20_000, save_path="./checkpoints/", name_prefix="ppo_agent"
    )

    eval_env = _build_env(eval_features, window=window)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/eval_logs/",
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    model.save(out_path)
    print(f"✅ Training complete. Model saved to {out_path}")
    return model
