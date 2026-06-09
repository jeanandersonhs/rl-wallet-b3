"""
PortfolioEnv - Ambiente de Reinforcement Learning para gestao de portfolio (B3).
"""

import itertools

import numpy as np
import pandas as pd


class PortfolioEnv:
    """
    Ambiente tabular simplificado para gestao de portfolio.

    O estado contem apenas retornos diarios dos ativos e pesos atuais da
    carteira. A recompensa prioriza retorno liquido, com regularizadores
    pequenos para diversificacao e concentracao.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,
        weight_delta: float = 0.10,
        alpha_diversification: float = 0.001,
        beta_concentration: float = 0.002,
        concentration_threshold: float = 0.60,
    ):
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.weight_delta = weight_delta
        self.alpha_diversification = alpha_diversification
        self.beta_concentration = beta_concentration
        self.concentration_threshold = concentration_threshold

        self.asset_names = ["ITUB4", "BOVA11", "BBAS3"]
        self.n_assets = len(self.asset_names)

        self.prices = df[self.asset_names].values.astype(np.float64)
        self.returns = df[
            ["retorno_ITUB4", "retorno_BOVA11", "retorno_BBAS3"]
        ].values.astype(np.float64)
        self.n_steps = len(df)

        self._action_deltas = np.array(
            list(itertools.product([-1, 0, 1], repeat=self.n_assets)),
            dtype=np.float64,
        )
        self.n_actions = len(self._action_deltas)
        self.action_space_size = self.n_actions
        self.observation_space_size = 6

        self.current_step = 0
        self.weights = np.zeros(self.n_assets)
        self.balance = 0.0
        self.portfolio_value = 0.0
        self.peak_value = 0.0

    def reset(self) -> np.ndarray:
        """Reinicia o ambiente e retorna o estado inicial."""
        self.current_step = 0
        self.weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance

        return self._get_observation()

    def _decode_action(self, action_id: int) -> np.ndarray:
        """Converte um indice de acao em deltas de peso."""
        return self._action_deltas[action_id] * self.weight_delta

    def _apply_weights(self, deltas: np.ndarray) -> tuple[np.ndarray, float]:
        """Aplica deltas aos pesos, normaliza a carteira e calcula custo."""
        new_weights = self.weights + deltas
        new_weights = np.clip(new_weights, 0.0, None)

        weight_sum = np.sum(new_weights)
        if weight_sum < 1e-10:
            new_weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        else:
            new_weights = new_weights / weight_sum

        turnover = np.sum(np.abs(new_weights - self.weights))
        cost = self.transaction_cost * turnover

        return new_weights, cost

    def _get_observation(self) -> np.ndarray:
        """
        Retorna estado 6D: retornos diarios dos 3 ativos + pesos atuais.
        """
        daily_returns = self.returns[self.current_step]
        return np.concatenate([daily_returns, self.weights])

    def _calculate_reward(self, portfolio_return: float, cost: float) -> float:
        """
        Recompensa: retorno liquido com regularizadores de carteira.

        reward = retorno_portfolio - custo
                 + alpha_diversification * (1 - HHI)
                 - beta_concentration * max(0, max_weight - threshold)
        """
        hhi = np.sum(self.weights ** 2)
        max_weight = np.max(self.weights)
        bonus_diversification = self.alpha_diversification * (1.0 - hhi)
        penalty_concentration = self.beta_concentration * max(
            0.0, max_weight - self.concentration_threshold
        )

        return (
            portfolio_return
            - cost
            + bonus_diversification
            - penalty_concentration
        )

    def step(self, action: int):
        """
        Executa uma transicao do MDP.

        Returns
        -------
        tuple[np.ndarray, float, bool, dict]
            (observacao, recompensa, done, info)
        """
        deltas = self._decode_action(action)
        new_weights, cost = self._apply_weights(deltas)
        self.weights = new_weights

        self.current_step += 1

        asset_returns = self.returns[self.current_step]
        portfolio_return = float(np.dot(self.weights, asset_returns))

        self.portfolio_value *= 1.0 + portfolio_return - cost
        self.peak_value = max(self.peak_value, self.portfolio_value)

        reward = float(self._calculate_reward(portfolio_return, cost))
        done = self.current_step >= self.n_steps - 1
        obs = self._get_observation()

        drawdown = (
            (self.peak_value - self.portfolio_value) / self.peak_value
            if self.peak_value > 0 else 0.0
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
            "portfolio_return": portfolio_return,
            "cost": cost,
            "drawdown": drawdown,
            "step": self.current_step,
        }

        return obs, reward, done, info

    def render(self, mode: str = "human"):
        """Imprime estado atual no console."""
        drawdown = (
            (self.peak_value - self.portfolio_value) / self.peak_value
            if self.peak_value > 0 else 0.0
        )

        print(
            f"Step {self.current_step:4d} | "
            f"Value: R$ {self.portfolio_value:,.2f} | "
            f"DD: {drawdown * 100:.1f}%"
        )
        for i, name in enumerate(self.asset_names):
            print(f"  {name}: {self.weights[i] * 100:.1f}%", end="")
        print()
