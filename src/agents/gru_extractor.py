import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    Extracteur GRU bi-branche :
      - GRU marché (multi-actifs)
      - GRU macro (facteurs globaux)
      - MLP sur allocation précédente
      - Fusion finale stabilisée
    Compatible avec :
        market: (B, W, N, F)
        macro:  (B, W, F_macro)
        alloc_prev: (B, N+1)
    """
    def __init__(self, observation_space: spaces.Dict,
                 features_dim: int = 256,
                 hidden_size: int = 128,
                 macro_hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__(observation_space, features_dim)

        # --- Dimensions extraites des spaces
        self.window = observation_space.spaces["market"].shape[0]
        self.n_assets = observation_space.spaces["market"].shape[1]
        self.n_feat = observation_space.spaces["market"].shape[2]
        self.f_macro = observation_space.spaces["macro"].shape[1]
        self.alloc_dim = observation_space.spaces["alloc_prev"].shape[0]
        # === Dimensions projetées équilibrées
        self.market_dim = features_dim // 3
        self.macro_dim = features_dim // 3
        # Ajustement automatique pour que la somme tombe juste
        self.alloc_proj_dim = features_dim - (self.market_dim + self.macro_dim)

        # === GRU marché
        self.gru_market = nn.GRU(self.n_assets * self.n_feat, hidden_size, batch_first=True)
        self.norm_market = nn.LayerNorm(hidden_size)

        # === GRU macro
        self.gru_macro = nn.GRU(self.f_macro, macro_hidden, batch_first=True)
        self.norm_macro = nn.LayerNorm(macro_hidden)

        # === MLPs individuels
        self.fc_market = nn.Sequential(
            nn.Linear(hidden_size, self.market_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_macro = nn.Sequential(
            nn.Linear(macro_hidden, self.macro_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_alloc = nn.Sequential(
            nn.Linear(self.alloc_dim, self.alloc_proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        # === Fusion finale
        concat_dim = self.market_dim + self.macro_dim + self.alloc_proj_dim

        self.fc_out = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Dropout(dropout)
        )

    def forward(self, obs):
        # --- Market sequence
        market = obs["market"]  # (B, W, N, F)
        b, w, n, f = market.shape
        market = market.view(b, w, n * f)
        _, h_market = self.gru_market(market)
        h_market = self.norm_market(h_market.squeeze(0))
        feat_market = self.fc_market(h_market)

        # --- Macro sequence
        macro = obs["macro"]  # (B, W, F_macro)
        _, h_macro = self.gru_macro(macro)
        h_macro = self.norm_macro(h_macro.squeeze(0))
        feat_macro = self.fc_macro(h_macro)

        # --- Previous allocation
        alloc_prev = obs["alloc_prev"]  # (B, N+1)
        feat_alloc = self.fc_alloc(alloc_prev)

        # --- Fusion des 3 branches
        fused = th.cat([feat_market, feat_macro, feat_alloc], dim=-1)
        out = self.fc_out(fused)
        return out
