import argparse
from src.train.train import run_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns", type=str, default="data/processed/returns.csv")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--out", type=str, default="experiments/ppo_gru")
    args = parser.parse_args()

    run_train(csv_returns_path=args.returns, total_timesteps=args.timesteps, window=args.window, n_envs=args.n_envs, out_path=args.out)
