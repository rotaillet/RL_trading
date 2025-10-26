import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    Extracteur de features basé sur un GRU avec régularisation.
    - Dropout pour éviter le surapprentissage
    - LayerNorm pour stabiliser les activations
    Compatible avec (batch, window, n_assets, n_features)
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128, hidden_size: int = 64, dropout: float = 0.4):
        super(GRUFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Dimensions observation
        self.window, self.n_assets, self.n_feat = observation_space.spaces["market"].shape
        self.input_dim = self.n_assets * self.n_feat
        self.alloc_dim = observation_space.spaces["alloc_prev"].shape[0]

        # GRU
        self.gru = nn.GRU(self.input_dim, hidden_size, batch_first=True)
        self.norm_gru = nn.LayerNorm(hidden_size)

        # Couches denses avec dropout
        self.fc_market = nn.Sequential(
            nn.Linear(hidden_size, features_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_alloc = nn.Sequential(
            nn.Linear(self.alloc_dim, features_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion finale
        self.fc_out = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)  # stabilisation finale
        )

    def forward(self, observations):
        market = observations["market"]   # (batch, window, n_assets, n_feat)
        b, w, n, f = market.shape
        market = market.view(b, w, n*f)   # aplatis N*F → GRU input

        alloc_prev = observations["alloc_prev"]  # (batch, N+1)

        # GRU encode la séquence
        _, h = self.gru(market)  # h: (1, batch, hidden_size)
        h = self.norm_gru(h.squeeze(0))  # (batch, hidden_size)

        market_feat = self.fc_market(h)
        alloc_feat = self.fc_alloc(alloc_prev)

        x = th.cat([market_feat, alloc_feat], dim=-1)
        return self.fc_out(x)
