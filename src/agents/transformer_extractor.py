import math
from typing import Optional

import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-2)
        return x + self.pe[:length]


class HierarchicalTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        window: int,
        d_model: int = 128,
        time_heads: int = 4,
        asset_heads: int = 4,
        time_layers: int = 2,
        asset_layers: int = 1,
        dropout: float = 0.1,
        max_assets: int = 512,
    ):
        super().__init__()
        self.window = window
        self.max_assets = max_assets
        self.feature_proj = nn.Linear(n_features, d_model)

        encoder_layer_time = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=time_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer_time, num_layers=time_layers)
        self.temporal_pos = SinusoidalPositionEncoding(d_model, max_len=window)

        encoder_layer_asset = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=asset_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.asset_encoder = nn.TransformerEncoder(encoder_layer_asset, num_layers=asset_layers)
        self.asset_pos = SinusoidalPositionEncoding(d_model, max_len=max_assets)

        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_assets, n_features)
        b, w, n, f = x.shape
        x = self.feature_proj(x)  # (b, w, n, d)
        x = x.permute(0, 2, 1, 3).reshape(b * n, w, -1)
        x = self.temporal_pos(x)
        x = self.temporal_encoder(x)
        x = x.mean(dim=1)  # (b * n, d)
        x = x.view(b, n, -1)
        x = self.asset_pos(x)
        x = self.asset_encoder(x)
        x = x.mean(dim=1)
        return self.norm_out(x)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        d_model: int = 128,
        time_heads: int = 4,
        asset_heads: int = 4,
        time_layers: int = 2,
        asset_layers: int = 1,
        dropout: float = 0.1,
        max_assets: int = 512,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        market_space = observation_space.spaces["market"]
        self.window, self.n_assets, self.n_features = market_space.shape
        self.alloc_dim = observation_space.spaces["alloc_prev"].shape[0]

        self.encoder = HierarchicalTransformerEncoder(
            n_features=self.n_features,
            window=self.window,
            d_model=d_model,
            time_heads=time_heads,
            asset_heads=asset_heads,
            time_layers=time_layers,
            asset_layers=asset_layers,
            dropout=dropout,
            max_assets=max_assets,
        )

        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location="cpu")
            encoder_state = state.get("encoder", state)
            missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
            if missing:
                print(f"[TransformerFeatureExtractor] Missing keys when loading pretrained encoder: {missing}")
            if unexpected:
                print(f"[TransformerFeatureExtractor] Unexpected keys when loading pretrained encoder: {unexpected}")

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.alloc_mlp = nn.Sequential(
            nn.Linear(self.alloc_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        market = observations["market"]
        alloc_prev = observations["alloc_prev"]

        encoded = self.encoder(market)
        alloc = self.alloc_mlp(alloc_prev)
        fused = torch.cat([encoded, alloc], dim=-1)
        return self.proj(fused)
