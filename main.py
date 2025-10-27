import argparse
from typing import Sequence

from src.pipeline.full_pipeline import (
    DEFAULT_PRETRAIN_TICKERS,
    DEFAULT_RL_TICKERS,
    PipelineConfig,
    run_full_pipeline,
)
from src.train.train import run_train


def _parse_tickers(raw: Sequence[str]) -> Sequence[str]:
    if isinstance(raw, str):
        return [raw]
    return list(dict.fromkeys(raw))


def main():
    parser = argparse.ArgumentParser(description="RL trading pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train PPO agent only")
    train_parser.add_argument("--timesteps", type=int, default=50_000)
    train_parser.add_argument("--window", type=int, default=20)
    train_parser.add_argument("--n-envs", type=int, default=4)
    train_parser.add_argument("--out", type=str, default="experiments/ppo_gru")
    train_parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Universe of tickers to load when generating features on the fly.",
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run pretraining + PPO fine-tuning")
    pipeline_parser.add_argument("--total-timesteps", type=int, default=200_000)
    pipeline_parser.add_argument("--window", type=int, default=20)
    pipeline_parser.add_argument("--n-envs", type=int, default=4)
    pipeline_parser.add_argument("--pretrain-epochs", type=int, default=25)
    pipeline_parser.add_argument("--pretrain-batch", type=int, default=64)
    pipeline_parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    pipeline_parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    pipeline_parser.add_argument("--encoder-checkpoint", type=str, default="checkpoints/pretrained_encoder.pt")
    pipeline_parser.add_argument("--policy-checkpoint", type=str, default="experiments/ppo_transformer")
    pipeline_parser.add_argument("--freeze-encoder", action="store_true")
    pipeline_parser.add_argument(
        "--pretrain-tickers", nargs="*", default=DEFAULT_PRETRAIN_TICKERS, help="Tickers used during encoder pretraining",
    )
    pipeline_parser.add_argument(
        "--rl-tickers", nargs="*", default=DEFAULT_RL_TICKERS, help="Tickers used for PPO fine-tuning",
    )
    pipeline_parser.add_argument("--pretrain-train-start", type=str, default="2004-01-01")
    pipeline_parser.add_argument("--pretrain-train-end", type=str, default="2018-12-31")
    pipeline_parser.add_argument("--pretrain-val-start", type=str, default="2019-01-01")
    pipeline_parser.add_argument("--pretrain-val-end", type=str, default="2020-12-31")
    pipeline_parser.add_argument("--rl-train-start", type=str, default="2010-01-01")
    pipeline_parser.add_argument("--rl-train-end", type=str, default="2018-12-31")
    pipeline_parser.add_argument("--rl-val-start", type=str, default="2019-01-01")
    pipeline_parser.add_argument("--rl-val-end", type=str, default="2021-12-31")
    pipeline_parser.add_argument("--rl-test-start", type=str, default="2022-01-01")
    pipeline_parser.add_argument("--rl-test-end", type=str, default="2024-12-31")

    args = parser.parse_args()

    if args.command == "train":
        tickers = _parse_tickers(args.tickers) if args.tickers else None
        run_train(
            total_timesteps=args.timesteps,
            window=args.window,
            n_envs=args.n_envs,
            out_path=args.out,
            tickers=tickers,
        )
    elif args.command == "pipeline":
        config = PipelineConfig(
            pretrain_tickers=_parse_tickers(args.pretrain_tickers),
            rl_tickers=_parse_tickers(args.rl_tickers),
            window=args.window,
            pretrain_train_period=(args.pretrain_train_start, args.pretrain_train_end),
            pretrain_val_period=(args.pretrain_val_start, args.pretrain_val_end),
            rl_train_period=(args.rl_train_start, args.rl_train_end),
            rl_val_period=(args.rl_val_start, args.rl_val_end),
            rl_test_period=(args.rl_test_start, args.rl_test_end),
            encoder_checkpoint=args.encoder_checkpoint,
            policy_checkpoint=args.policy_checkpoint,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_batch_size=args.pretrain_batch,
            pretrain_lr=args.pretrain_lr,
            pretrain_weight_decay=args.pretrain_weight_decay,
            freeze_encoder=args.freeze_encoder,
        )
        metrics = run_full_pipeline(config)
        for split, values in metrics.items():
            print(f"=== {split.upper()} METRICS ===")
            for key, value in values.items():
                print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
