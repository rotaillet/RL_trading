import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.agents.transformer_extractor import HierarchicalTransformerEncoder


class MarketWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, window: int, horizon: int = 1):
        self.features = features.astype(np.float32)
        self.window = window
        self.horizon = horizon
        self.T = self.features.shape[0]
        self.n_assets = self.features.shape[1]

    def __len__(self) -> int:
        return max(0, self.T - self.window - self.horizon + 1)

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.window
        target_idx = end + self.horizon - 1
        x = self.features[start:end]
        y = self.features[target_idx, :, 0]
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class PretrainConfig:
    tickers: Sequence[str]
    train_period: Tuple[str, str]
    val_period: Tuple[str, str]
    window: int = 20
    horizon: int = 1
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    d_model: int = 128
    time_heads: int = 4
    asset_heads: int = 4
    time_layers: int = 2
    asset_layers: int = 1
    dropout: float = 0.1
    max_assets: int = 512
    save_path: str = "checkpoints/pretrained_encoder.pt"
    cache_dir: Optional[str] = "data/cache"
    feature_list: Iterable[str] = (
        "ret_log",
        "vol_20",
        "hl_range",
        "mom_21",
        "ma_ratio_20",
        "ma_ratio_50",
        "rsi_14",
        "atr_14",
    )


class EncoderPredictor(nn.Module):
    def __init__(self, encoder: HierarchicalTransformerEncoder, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        preds = self.head(latent).squeeze(-1)
        return preds


def build_dataloaders(
    train_features: np.ndarray,
    val_features: np.ndarray,
    window: int,
    horizon: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = MarketWindowDataset(train_features, window=window, horizon=horizon)
    val_ds = MarketWindowDataset(val_features, window=window, horizon=horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def pretrain_encoder(
    config: PretrainConfig,
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> str:
    device = torch.device(config.device)
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        train_features, val_features, config.window, config.horizon, config.batch_size
    )

    encoder = HierarchicalTransformerEncoder(
        n_features=train_features.shape[-1],
        window=config.window,
        d_model=config.d_model,
        time_heads=config.time_heads,
        asset_heads=config.asset_heads,
        time_layers=config.time_layers,
        asset_layers=config.asset_layers,
        dropout=config.dropout,
        max_assets=config.max_assets,
    )

    model = EncoderPredictor(encoder, config.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"[pretrain] epoch={epoch+1}/{config.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "encoder": model.encoder.state_dict(),
                "config": config,
            }

    if best_state is None:
        raise RuntimeError("Pretraining failed to produce a valid checkpoint")

    torch.save(best_state, config.save_path)
    print(f"✅ Pretrained encoder saved to {config.save_path}")
    return config.save_path
