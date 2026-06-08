"""
PortfolioEnv — Ambiente de Reinforcement Learning para Gestão de Portfólio (B3)

Interface compatível com OpenAI Gym (reset/step/render), sem dependência direta.

Ativos: ITUB4 (ação), BOVA11 (ETF Ibovespa), BBAS3 (ação)
Estado: vetor contínuo de 18 dimensões
Ações: 27 combinações discretas (comprar/manter/vender por ativo)
Recompensa: Sharpe Ratio modificado + bônus/penalidades

Referência completa: documentacao.txt, seções 3 e 6.
"""

import itertools
import numpy as np
import pandas as pd


class PortfolioEnv:
    """
    Ambiente de Reinforcement Learning para gestão de portfólio.

    Métodos públicos:
        reset()       → np.ndarray (estado inicial)
        step(action)  → (obs, reward, done, info)
        render()      → imprime estado atual no console
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,
        window_size: int = 20,
        weight_delta: float = 0.10,
        lambda_transaction: float = 0.1,
        alpha_benchmark: float = 0.5,
        alpha_diversification: float = 0.2,
        beta_drawdown: float = 2.0,
        beta_concentration: float = 1.0,
        drawdown_threshold: float = 0.10,
        concentration_threshold: float = 0.60,
    ):
        """
        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame com colunas: data, ITUB4, BOVA11, BBAS3,
            retorno_ITUB4, retorno_BOVA11, retorno_BBAS3,
            CDI_decimal, retorno_acima_cdi_ITUB4,
            retorno_acima_cdi_BOVA11, retorno_acima_cdi_BBAS3
        initial_balance : float
            Capital inicial em BRL.
        transaction_cost : float
            Custo por unidade de turnover (c na fórmula).
        window_size : int
            Janela para rolling mean/std (dias).
        weight_delta : float
            Δw aplicado por ação de compra/venda.
        lambda_transaction : float
            λ — coeficiente de penalidade de transação na recompensa.
        alpha_benchmark : float
            α₁ — bônus por superar BOVA11.
        alpha_diversification : float
            α₂ — bônus por diversificação (via HHI).
        beta_drawdown : float
            β₁ — penalidade por drawdown excessivo.
        beta_concentration : float
            β₂ — penalidade por concentração em um único ativo.
        drawdown_threshold : float
            Limite de drawdown antes da penalidade (10%).
        concentration_threshold : float
            Limite de peso máximo antes da penalidade (60%).
        """
        # --- Configuração ---
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        self.weight_delta = weight_delta

        # Hiperparâmetros de recompensa
        self.lambda_transaction = lambda_transaction
        self.alpha_benchmark = alpha_benchmark
        self.alpha_diversification = alpha_diversification
        self.beta_drawdown = beta_drawdown
        self.beta_concentration = beta_concentration
        self.drawdown_threshold = drawdown_threshold
        self.concentration_threshold = concentration_threshold

        # --- Dados do mercado (extraídos do DataFrame) ---
        self.asset_names = ["ITUB4", "BOVA11", "BBAS3"]
        self.n_assets = len(self.asset_names)

        self.prices = df[self.asset_names].values.astype(np.float64)

        self.returns = df[
            ["retorno_ITUB4", "retorno_BOVA11", "retorno_BBAS3"]
        ].values.astype(np.float64)

        self.cdi = df["CDI_decimal"].values.astype(np.float64)

        self.returns_above_cdi = df[
            ["retorno_acima_cdi_ITUB4", "retorno_acima_cdi_BOVA11",
             "retorno_acima_cdi_BBAS3"]
        ].values.astype(np.float64)

        # Retorno do BOVA11 isolado (benchmark)
        self.bova11_returns = df["retorno_BOVA11"].values.astype(np.float64)

        self.n_steps = len(df)

        # --- Espaço de ações ---
        # Lookup table: cada linha é (delta_ITUB4, delta_BOVA11, delta_BBAS3)
        # com valores em {-1, 0, +1}, depois multiplicados por weight_delta
        self._action_deltas = np.array(
            list(itertools.product([-1, 0, 1], repeat=self.n_assets)),
            dtype=np.float64
        )  # shape (27, 3)
        self.n_actions = len(self._action_deltas)  # 27
        self.action_space_size = self.n_actions
        self.observation_space_size = 18

        # --- Estado interno (inicializado no reset) ---
        self.current_step = 0
        self.weights = np.zeros(self.n_assets)
        self.balance = 0.0
        self.portfolio_value = 0.0
        self.peak_value = 0.0
        self.portfolio_returns_history = []

    def reset(self) -> np.ndarray:
        """
        Reinicia o ambiente para o estado inicial.

        Returns
        -------
        np.ndarray
            Vetor de observação (shape 18).
        """
        self.current_step = self.window_size  # pula primeiros dias para janela
        self.weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.portfolio_returns_history = []

        return self._get_observation()

    def _decode_action(self, action_id: int) -> np.ndarray:
        """
        Converte um inteiro [0, 26] em vetor de deltas de peso.

        Parameters
        ----------
        action_id : int
            Índice da ação (0 a 26).

        Returns
        -------
        np.ndarray
            Vetor de deltas, shape (3,). Ex: [-0.10, 0.0, +0.10]
        """
        return self._action_deltas[action_id] * self.weight_delta

   
    def _apply_weights(self, deltas: np.ndarray):
        """
        Aplica deltas aos pesos atuais, garante restrições, calcula custo.

        Parameters
        ----------
        deltas : np.ndarray
            Vetor de deltas de peso, shape (3,).

        Returns
        -------
        tuple[np.ndarray, float]
            (novos_pesos normalizados, custo_de_transação)
        """
        new_weights = self.weights + deltas

        # Long-only: sem posições negativas
        new_weights = np.clip(new_weights, 0.0, None)

        # Se todos os pesos forem zero (edge case), reset para iguais
        weight_sum = np.sum(new_weights)
        if weight_sum < 1e-10:
            new_weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        else:
            # Normaliza para Σ w_i = 1.0 (fully invested)
            new_weights = new_weights / weight_sum

        # Custo de transação: c · Σ |w_new - w_old|
        turnover = np.sum(np.abs(new_weights - self.weights))
        cost = self.transaction_cost * turnover

        return new_weights, cost

  
    def _get_observation(self) -> np.ndarray:
        """
        Monta o vetor de estado S(t) com 18 dimensões.

        Composição:
            [0:3]   Retornos diários dos 3 ativos
            [3:6]   Média (window_size) dos retornos por ativo
            [6:9]   Desvio-padrão dos retornos por ativo
            [9:12]  Retorno acima do CDI por ativo
            [12]    CDI do dia
            [13:16] Pesos atuais do portfólio
            [16]    Drawdown acumulado
            [17]    Volatilidade realizada do portfólio

        Returns
        -------
        np.ndarray
            Vetor de observação, shape (18,).
        """
        t = self.current_step

        # 1. Retornos do dia (3)
        daily_returns = self.returns[t]

        # 2. Rolling mean e std dos retornos (3 + 3 = 6)
        start = max(0, t - self.window_size + 1)
        window = self.returns[start: t + 1]
        rolling_mean = np.mean(window, axis=0)
        rolling_std = np.std(window, axis=0)

        # 3. Excesso sobre CDI (3)
        excess_cdi = self.returns_above_cdi[t]

        # 4. CDI do dia (1)
        cdi = np.array([self.cdi[t]])

        # 5. Pesos atuais do portfólio (3)
        weights = self.weights

        # 6. Drawdown acumulado (1)
        if self.peak_value > 0:
            drawdown = np.array([
                (self.peak_value - self.portfolio_value) / self.peak_value
            ])
        else:
            drawdown = np.array([0.0])

        # 7. Volatilidade realizada do portfólio (1)
        if len(self.portfolio_returns_history) >= 2:
            recent = self.portfolio_returns_history[-self.window_size:]
            port_vol = np.array([np.std(recent)])
        else:
            port_vol = np.array([0.0])

        state = np.concatenate([
            daily_returns,    # 3
            rolling_mean,     # 3
            rolling_std,      # 3
            excess_cdi,       # 3
            cdi,              # 1
            weights,          # 3
            drawdown,         # 1
            port_vol,         # 1
        ])  # total = 18

        return state

    # ─────────────────────────────────────────────────────────────────────
    # 6. CALCULATE REWARD
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_reward(
        self,
        portfolio_return: float,
        cost: float,
        bova11_return: float
    ) -> float:
        """
        Calcula a recompensa completa do timestep.

        R(t) = sharpe_instantâneo
             + α₁ · max(0, rₚ − r_BOVA11)       (bônus benchmark)
             + α₂ · (1 − HHI)                    (bônus diversificação)
             − β₁ · max(0, dd − threshold_dd)     (penalidade drawdown)
             − β₂ · max(0, max(w) − threshold_c)  (penalidade concentração)
             − λ  · custo_transação               (penalidade transação)

        Parameters
        ----------
        portfolio_return : float
            Retorno do portfólio no timestep.
        cost : float
            Custo de transação incorrido.
        bova11_return : float
            Retorno do BOVA11 no mesmo timestep (benchmark).

        Returns
        -------
        float
            Recompensa total.
        """
        t = self.current_step
        cdi_daily = self.cdi[t]

        # --- Sharpe instantâneo ---
        # Volatilidade do portfólio (mínimo 1e-8 para evitar divisão por zero)
        if len(self.portfolio_returns_history) >= 2:
            recent = self.portfolio_returns_history[-self.window_size:]
            sigma = max(np.std(recent), 1e-8)
        else:
            sigma = 1e-8

        sharpe = (portfolio_return - cdi_daily) / sigma

        # --- Bônus: superar benchmark BOVA11 ---
        bonus_benchmark = self.alpha_benchmark * max(
            0.0, portfolio_return - bova11_return
        )

        # --- Bônus: diversificação (HHI) ---
        # HHI = Σ wᵢ²; quanto menor, mais diversificado
        # HHI = 1/N para diversificação perfeita, HHI = 1 para concentração total
        hhi = np.sum(self.weights ** 2)
        bonus_diversification = self.alpha_diversification * (1.0 - hhi)

        # --- Penalidade: drawdown excessivo ---
        if self.peak_value > 0:
            dd = (self.peak_value - self.portfolio_value) / self.peak_value
        else:
            dd = 0.0
        penalty_drawdown = self.beta_drawdown * max(
            0.0, dd - self.drawdown_threshold
        )

        # --- Penalidade: concentração excessiva ---
        penalty_concentration = self.beta_concentration * max(
            0.0, np.max(self.weights) - self.concentration_threshold
        )

        # --- Penalidade: custo de transação ---
        penalty_transaction = self.lambda_transaction * cost

        # --- Recompensa total ---
        reward = (
            sharpe
            + bonus_benchmark
            + bonus_diversification
            - penalty_drawdown
            - penalty_concentration
            - penalty_transaction
        )

        return reward

    # ─────────────────────────────────────────────────────────────────────
    # 7. STEP
    # ─────────────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Executa uma transição do MDP.

        Parameters
        ----------
        action : int
            Índice da ação (0 a 26).

        Returns
        -------
        tuple[np.ndarray, float, bool, dict]
            (observação, recompensa, done, info)
        """
        # 1. Decodifica ação em deltas de peso
        deltas = self._decode_action(action)

        # 2. Aplica deltas e calcula custo de transação
        new_weights, cost = self._apply_weights(deltas)
        self.weights = new_weights

        # 3. Avança um dia no mercado
        self.current_step += 1

        # 4. Calcula retorno do portfólio: rₚ = Σ wᵢ · rᵢ
        asset_returns = self.returns[self.current_step]
        portfolio_return = np.dot(self.weights, asset_returns)

        # 5. Atualiza valor do portfólio (desconta custo de transação)
        self.portfolio_value *= (1.0 + portfolio_return - cost)

        # 6. Atualiza pico (para cálculo de drawdown)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # 7. Registra retorno na história
        self.portfolio_returns_history.append(portfolio_return)

        # 8. Calcula recompensa
        bova11_return = self.bova11_returns[self.current_step]
        reward = self._calculate_reward(portfolio_return, cost, bova11_return)

        # 9. Verifica condição de término
        done = self.current_step >= self.n_steps - 1

        # 10. Monta observação do novo estado
        obs = self._get_observation()

        # 11. Info dict com métricas úteis para análise
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
            "portfolio_return": portfolio_return,
            "cost": cost,
            "drawdown": (
                (self.peak_value - self.portfolio_value) / self.peak_value
                if self.peak_value > 0 else 0.0
            ),
            "step": self.current_step,
        }

        return obs, reward, done, info

    # ─────────────────────────────────────────────────────────────────────
    # 8. RENDER
    # ─────────────────────────────────────────────────────────────────────

    def render(self, mode: str = "human"):
        """
        Imprime o estado atual do ambiente no console.

        Parameters
        ----------
        mode : str
            Modo de renderização ('human' para console).
        """
        if self.peak_value > 0:
            dd = (self.peak_value - self.portfolio_value) / self.peak_value
        else:
            dd = 0.0

        print(
            f"Step {self.current_step:4d} | "
            f"Value: R$ {self.portfolio_value:,.2f} | "
            f"DD: {dd * 100:.1f}%"
        )
        for i, name in enumerate(self.asset_names):
            print(f"  {name}: {self.weights[i] * 100:.1f}%", end="")
        print()