import gym
import numpy as np
from gym import spaces

class PortfolioEnv(gym.Env):
    """
    Environnement de trading multi-actifs avec features supplémentaires.

    Hypothèses importantes :
    - Observations = dict:
        - "market": fenêtre (window, N, F) où la dimension F[0] = log-return journalier de l'actif
        - "alloc_prev": poids précédents (N actifs + 1 cash)
    - Actions = poids (N actifs + 1 cash), contraints à somme=1 via renormalisation
    - Mise à jour de la valeur du portefeuille en ARITHMÉTIQUE:
        value_t+1 = value_t * (1 + r_portefeuille - coût_tx)
      où r_portefeuille = dot(weights_actifs, returns_arithm) et
          returns_arithm = exp(logret) - 1
    - Turnover : pénalisation L1 du changement d'allocations
        - par défaut, SANS le cash (plus réaliste)
        - coût effectif = tx_cost * (0.5 * L1) (option activée par défaut)
    - Reward par défaut (stable/minimaliste) :
        reward = log(1 + r_portefeuille - coût_tx) - lambda_turn * L1
      -> on peut réactiver un terme de Sharpe instantané (rolling) et des bonus finaux
         (CAGR / MaxDrawdown) via des options.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, features: np.ndarray,features_macro: np.ndarray, window: int = 20, tx_cost: float = 0.003):
        super().__init__()
        self.features = features.astype(np.float32)   # (T, N, F) avec features[:, :, 0] = log-returns
        self.features_macro = features_macro.astype(np.float32)
        self.T,self.N, self.F = features.shape
        self.T2, self.F2 = features_macro.shape
        self.window = window
        self.tx_cost = tx_cost

        # Coefficients (valeurs par défaut raisonnables)
        self.lambda_turn = 0.01     # pénalisation du turnover (L1 brut)
        self.lambda_sharpe = 0.003   # poids du rolling Sharpe si activé
        self.lambda_cagr = 0.25      # bonus terminal CAGR si activé
        self.lambda_dd = 0.5         # pénalité terminale max drawdown si activé
        self.sharpe_window = 20

        # Options de comportement
        self.turnover_on_assets_only = True   # True: on ne pénalise pas le cash dans le L1
        self.half_turnover_cost = False        # True: coût = tx_cost * (0.5 * L1), sinon tx_cost * L1
        self.enable_rolling_sharpe = False    # False par défaut (à activer après stabilisation)
        self.enable_terminal_bonuses = False  # False par défaut (à activer après stabilisation)

        # Espaces Gym
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(self.window, self.N, self.F), dtype=np.float32),
            "macro": spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(self.window, self.F2), dtype=np.float32),
            "alloc_prev": spaces.Box(low=0.0, high=1.0,
                                     shape=(self.N + 1,), dtype=np.float32)
        })
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.N + 1,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.t = self.window
        self.value = 1.0
        self.start_value = self.value
        # full cash au début
        self.weights = np.concatenate([np.zeros(self.N, dtype=np.float32),
                                       np.array([1.0], dtype=np.float32)])
        self.history_values = [float(self.value)]
        self.daily_returns_history = []  # on stocke des log-returns nets pour la reward/diagnostic
        self.peak_value = float(self.value)
        self.max_drawdown = 0.0
        return self._get_obs()

    def _get_obs(self):
        market_obs = self.features[self.t - self.window:self.t]  # (window, N, F)
        macro_obs = self.features_macro[self.t - self.window:self.t]
        return {
            "market": market_obs,
            "macro": macro_obs,
            "alloc_prev": self.weights.astype(np.float32)
        }

    def _update_drawdown(self):
        if self.value > self.peak_value:
            self.peak_value = float(self.value)
        dd = (self.peak_value - self.value) / (self.peak_value + 1e-12)
        self.max_drawdown = float(max(self.max_drawdown, dd))

    def step(self, action):
        # --- Normalisation de l'action en simplex ---
        a = np.clip(action, 1e-8, 1.0)
        a = a.astype(np.float32)
        a /= a.sum() + 1e-8

        # --- Log-returns des actifs (dimension F[0]) ---
        daily_logret_assets = self.features[self.t, :, 0]            # shape (N,)
        daily_ret_assets_lin = np.exp(daily_logret_assets) - 1.0     # arithmétique

        # --- Turnover L1 (par défaut uniquement sur les actifs) ---
        if self.turnover_on_assets_only:
            prev_w = self.weights[:-1]
            new_w  = a[:-1]
        else:
            prev_w = self.weights
            new_w  = a
        l1 = float(np.sum(np.abs(new_w - prev_w)))
        effective_turnover = 0.5 * l1 if self.half_turnover_cost else l1
        cost = float(self.tx_cost * effective_turnover)

        # --- Rendement du portefeuille (arithmétique) ---
        port_ret_lin = float(np.dot(a[:-1], daily_ret_assets_lin))

        # --- Mise à jour de la valeur (arithmétique) ---
        gross = 1.0 + port_ret_lin - cost
        gross = max(gross, 1e-8)  # garde la valeur > 0
        self.value *= gross

        # --- Log-return net du jour (pour reward + diag) ---
        net_daily_logret = float(np.log(gross))
        self.daily_returns_history.append(net_daily_logret)
        self._update_drawdown()

        # --- Reward minimaliste + options ---
        # pénalisation sur le L1 "brut" (sans 0.5) pour pousser à moins réallouer
        r_turn = - self.lambda_turn * l1
        reward = net_daily_logret + r_turn

        # (Option) Ajout d'un Sharpe instantané (rolling) borné
        if self.enable_rolling_sharpe and len(self.daily_returns_history) >= self.sharpe_window:
            window = np.array(self.daily_returns_history[-self.sharpe_window:], dtype=np.float32)
            mean = float(window.mean())
            std = float(window.std(ddof=0)) + 1e-12
            rolling_sharpe = np.clip((mean / std) * np.sqrt(252.0), -5.0, 5.0)
            reward += self.lambda_sharpe * float(rolling_sharpe)

        # --- Avancement ---
        self.weights = a
        self.t += 1
        done = bool(self.t >= self.T)

        # (Option) Bonus/penalité fin d'épisode
        if done and self.enable_terminal_bonuses:
            total_log_return = float(np.log(self.value / (self.start_value + 1e-12)))
            reward += self.lambda_cagr * total_log_return
            reward += - self.lambda_dd * float(self.max_drawdown)

        # --- Observation suivante + infos ---
        if not done:
            obs = self._get_obs()
        else:
            obs = {
                "market": np.zeros((self.window, self.N, self.F), dtype=np.float32),
                "macro": np.zeros((self.window, self.F2), dtype=np.float32),
                "alloc_prev": np.zeros(self.N + 1, dtype=np.float32),
            }

        self.history_values.append(float(self.value))
        info = {
            "portfolio_value": float(self.value),
            "turnover": float(l1),                 # L1 brut (monitoring)
            "tx_cost_effective": float(effective_turnover),  # 0.5*L1 si option activée
            "max_drawdown": float(self.max_drawdown),
            "weights": a.copy(),                   # <--- AJOUT
        }
        return obs, float(reward), done, info

    def render(self, mode="human"):
        print(f"t={self.t}, value={self.value:.6f}")

    def get_value_series(self):
        return np.array(self.history_values, dtype=np.float32)
