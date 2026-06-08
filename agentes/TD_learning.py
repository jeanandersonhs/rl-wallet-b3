"""
AgentTD — Agente SARSA (on-policy TD control) para Gestão de Portfólio (B3)

Algoritmo on-policy de Temporal Difference (TD) control.

Equação de atualização (SARSA):
    Q(s, a) ← Q(s, a) + α · [ r + γ · Q(s', a') − Q(s, a) ]
                                       ↑
                          a' = ação REALMENTE tomada em s' (on-policy)

Diferença vs Q-Learning:
    Q-Learning (off-policy):  usa max_a' Q(s', a')  →  aprende política ótima
    SARSA      (on-policy):   usa Q(s', a')          →  aprende sobre a política atual

O SARSA é mais conservador: como inclui a exploração (ε-greedy) no aprendizado,
ele penaliza ações arriscadas que a política exploratória pode tomar. Isso é
vantajoso em finanças onde exploração excessiva é custosa.

O estado contínuo (18 dimensões) é discretizado em bins para indexar
a Q-table (dicionário esparso via defaultdict).

Referência: documentacao.txt, seções 3, 4.2 e 9.
"""

import pickle
from collections import defaultdict

import numpy as np

from src.feature_engineering import compute_bins_from_simulation, discretize_state


class AgentTD:
    """
    Agente SARSA (on-policy TD control) para gestão de portfólio.

    Parameters
    ----------
    env : PortfolioEnv
        Ambiente de RL (usado para calcular bins e obter n_actions).
    n_bins : int
        Número de bins para discretização de cada feature do estado.
    alpha : float
        Learning rate inicial.
    alpha_min : float
        Learning rate mínimo (limite inferior do decay).
    alpha_decay : float
        Fator de decay multiplicativo de α por episódio.
    gamma : float
        Fator de desconto γ (0.99 = horizonte longo).
    epsilon : float
        Probabilidade inicial de exploração (ε-greedy).
    epsilon_min : float
        ε mínimo.
    epsilon_decay : float
        Fator de decay multiplicativo de ε por episódio.
    n_sim_episodes : int
        Número de episódios de simulação aleatória para cálculo dos bins.
    seed : int
        Seed para reprodutibilidade.
    """

    def __init__(
        self,
        env,
        n_bins: int = 5,
        alpha: float = 0.1,
        alpha_min: float = 0.01,
        alpha_decay: float = 0.995,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        n_sim_episodes: int = 10,
        seed: int = 42,
    ):
        # --- Hiperparâmetros ---
        self.n_actions = env.n_actions          # 27 ações
        self.n_bins = n_bins
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.gamma = gamma                      # Fator de desconto (Bellman)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # --- Discretização ---
        # Calcula bins executando episódios aleatórios no ambiente de treino
        # para capturar a distribuição real dos estados (incluindo features
        # dinâmicas como drawdown, volatilidade e pesos do portfólio)
        print(f"[SARSA] Calculando bins de discretização "
              f"({n_sim_episodes} episódios aleatórios)...")
        self.bins = compute_bins_from_simulation(
            env, n_bins=n_bins, n_episodes=n_sim_episodes, seed=seed
        )
        print(f"[SARSA] Bins calculados para {len(self.bins)} features.")

        # --- Q-Table ---
        # Dicionário esparso: apenas estados visitados são armazenados.
        # Estado nunca visitado → vetor de zeros (sem viés inicial).
        # Chave: tupla discreta (ex: (2, 0, 4, 1, 3, ...))
        # Valor: np.array de Q-values, shape (27,)
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

    # ─────────────────────────────────────────────────────────────────────
    # DISCRETIZAÇÃO
    # ─────────────────────────────────────────────────────────────────────

    def discretize(self, state: np.ndarray) -> tuple:
        """
        Converte estado contínuo (18D) em tupla discreta.

        Parameters
        ----------
        state : np.ndarray
            Vetor de estado contínuo retornado pelo ambiente, shape (18,).

        Returns
        -------
        tuple[int, ...]
            Estado discreto — chave para a Q-table.
        """
        return discretize_state(state, self.bins)

    # ─────────────────────────────────────────────────────────────────────
    # POLÍTICA ε-GREEDY
    # ─────────────────────────────────────────────────────────────────────

    def choose_action(self, state: tuple) -> int:
        """
        Seleciona ação usando política ε-greedy.

        Com probabilidade ε: ação aleatória (EXPLORAÇÃO)
        Com probabilidade 1−ε: ação com maior Q-value (EXPLOITATION)

        Parameters
        ----------
        state : tuple
            Estado discreto (já discretizado).

        Returns
        -------
        int
            Índice da ação selecionada (0 a 26).
        """
        if self.rng.random() < self.epsilon:
            # Exploração: ação aleatória uniforme
            return self.rng.randint(0, self.n_actions)
        else:
            # Exploitation: ação gulosa (greedy)
            q_values = self.q_table[state]
            # Desempate aleatório quando há múltiplos máximos
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return self.rng.choice(best_actions)

    # ─────────────────────────────────────────────────────────────────────
    # ATUALIZAÇÃO Q — SARSA (ON-POLICY)
    # ─────────────────────────────────────────────────────────────────────

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        next_action: int,
        done: bool,
    ) -> float:
        """
        Atualiza Q(s, a) usando a equação SARSA (on-policy TD control).

        Q(s, a) ← Q(s, a) + α · [ r + γ · Q(s', a') − Q(s, a) ]

        Diferença crucial vs Q-Learning:
            Q-Learning: td_target = r + γ · max_a' Q(s', a')   (off-policy)
            SARSA:      td_target = r + γ · Q(s', a')          (on-policy)

        O a' no SARSA é a ação realmente escolhida pela política ε-greedy
        para o próximo estado, não a ação ótima teórica. Isso torna o SARSA
        mais conservador: ele "sabe" que às vezes vai explorar (e pode
        tomar ações ruins), e ajusta suas estimativas de acordo.

        Se done=True (episódio terminou), não há estado futuro:
            Q(s, a) ← Q(s, a) + α · [ r − Q(s, a) ]

        Parameters
        ----------
        state : tuple
            Estado discreto atual.
        action : int
            Ação tomada (0 a 26).
        reward : float
            Recompensa recebida do ambiente.
        next_state : tuple
            Próximo estado discreto (s').
        next_action : int
            Ação escolhida no próximo estado (a') — on-policy.
        done : bool
            Se o episódio terminou.

        Returns
        -------
        float
            TD error (δ) — útil para monitoramento do aprendizado.
        """
        # Estimativa atual
        current_q = self.q_table[state][action]

        # TD target SARSA: r + γ · Q(s', a')
        if done:
            # Estado terminal: sem recompensa futura
            td_target = reward
        else:
            # On-policy: usa Q do par (s', a') efetivamente escolhido
            # NÃO usa max — essa é a diferença fundamental vs Q-Learning
            next_q = self.q_table[next_state][next_action]
            td_target = reward + self.gamma * next_q

        # TD error: δ = target − estimativa atual
        # Se δ > 0: experiência MELHOR que esperado → Q sobe
        # Se δ < 0: experiência PIOR que esperado → Q desce
        td_error = td_target - current_q

        # Atualização incremental (SARSA update)
        self.q_table[state][action] += self.alpha * td_error

        return td_error

    # ─────────────────────────────────────────────────────────────────────
    # DECAY DE HIPERPARÂMETROS
    # ─────────────────────────────────────────────────────────────────────

    def decay_hyperparams(self) -> None:
        """
        Decai ε (exploração) e α (learning rate) ao final de cada episódio.

        O decay é multiplicativo:
            ε ← max(ε_min, ε × ε_decay)
            α ← max(α_min, α × α_decay)

        Isso permite:
            - Início: muita exploração (ε ≈ 1.0) e aprendizado rápido (α alto)
            - Final: quase pura exploitation (ε ≈ 0.01) e ajustes finos (α baixo)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    # ─────────────────────────────────────────────────────────────────────
    # TREINAMENTO — LOOP SARSA
    # ─────────────────────────────────────────────────────────────────────

    def train(
        self,
        env,
        n_episodes: int = 500,
        log_interval: int = 50,
    ) -> dict:
        """
        Treina o agente por N episódios usando SARSA (on-policy).

        Cada episódio = 1 passagem completa pelos dados de treino.

        Loop SARSA (diferença vs Q-Learning):
            1. Observa estado s(t) e discretiza
            2. Seleciona ação a(t) com ε-greedy      ← ANTES do loop
            3. Executa step → (s', r, done, info)
            4. Seleciona a'(t) com ε-greedy em s'    ← ANTES do update
            5. Atualiza Q(s, a) com Q(s', a')         ← on-policy
            6. s ← s', a ← a'
        Ao final do episódio: decai ε e α.

        Parameters
        ----------
        env : PortfolioEnv
            Ambiente de RL com dados de treino.
        n_episodes : int
            Número de episódios de treinamento (default: 500).
        log_interval : int
            Intervalo de episódios para logging (default: 50).

        Returns
        -------
        dict
            Histórico de treinamento com chaves:
            - 'episode_rewards': lista de reward total por episódio
            - 'episode_portfolio_values': valor final do portfólio por episódio
            - 'episode_epsilons': ε ao final de cada episódio
            - 'episode_alphas': α ao final de cada episódio
            - 'episode_td_errors': TD error médio por episódio
            - 'n_states_visited': nº de estados únicos na Q-table
        """
        history = {
            "episode_rewards": [],
            "episode_portfolio_values": [],
            "episode_epsilons": [],
            "episode_alphas": [],
            "episode_td_errors": [],
        }

        print(f"\n{'='*60}")
        print(f"  SARSA (TD) TREINAMENTO — {n_episodes} episódios")
        print(f"  α={self.alpha}, γ={self.gamma}, ε={self.epsilon}, "
              f"bins={self.n_bins}")
        print(f"{'='*60}\n")

        for episode in range(1, n_episodes + 1):
            # Reset do ambiente — início de novo episódio
            state_continuous = env.reset()
            state = self.discretize(state_continuous)

            # SARSA: escolhe a PRIMEIRA ação antes de entrar no loop
            action = self.choose_action(state)

            total_reward = 0.0
            total_td_error = 0.0
            n_steps = 0
            done = False

            while not done:
                # 1. Executa ação no ambiente
                next_state_continuous, reward, done, info = env.step(action)

                # 2. Discretiza próximo estado
                next_state = self.discretize(next_state_continuous)

                # 3. Escolhe a' no próximo estado (on-policy)
                #    Essa ação será usada TANTO no update QUANTO como
                #    ação do próximo timestep — essência do SARSA
                next_action = self.choose_action(next_state)

                # 4. Atualiza Q-table (SARSA: usa Q(s', a'), não max)
                td_error = self.update(
                    state, action, reward, next_state, next_action, done
                )

                # 5. Transição: s ← s', a ← a'
                state = next_state
                action = next_action
                total_reward += reward
                total_td_error += abs(td_error)
                n_steps += 1

            # Fim do episódio — decai hiperparâmetros
            self.decay_hyperparams()

            # Registra métricas
            history["episode_rewards"].append(total_reward)
            history["episode_portfolio_values"].append(info["portfolio_value"])
            history["episode_epsilons"].append(self.epsilon)
            history["episode_alphas"].append(self.alpha)
            history["episode_td_errors"].append(
                total_td_error / max(n_steps, 1)
            )

            # Logging periódico
            if episode % log_interval == 0 or episode == 1:
                avg_reward = np.mean(history["episode_rewards"][-log_interval:])
                print(
                    f"  Ep {episode:4d}/{n_episodes} | "
                    f"Reward: {total_reward:8.2f} | "
                    f"Avg({log_interval}): {avg_reward:8.2f} | "
                    f"Value: R$ {info['portfolio_value']:,.2f} | "
                    f"ε: {self.epsilon:.4f} | "
                    f"α: {self.alpha:.4f} | "
                    f"Q-states: {len(self.q_table):,}"
                )

        # Resumo final
        history["n_states_visited"] = len(self.q_table)

        print(f"\n{'='*60}")
        print(f"  TREINAMENTO SARSA CONCLUÍDO")
        print(f"  Estados visitados: {len(self.q_table):,}")
        print(f"  ε final: {self.epsilon:.4f}")
        print(f"  α final: {self.alpha:.4f}")
        print(f"  Reward médio (últimos 50): "
              f"{np.mean(history['episode_rewards'][-50:]):.2f}")
        print(f"{'='*60}\n")

        return history

    # ─────────────────────────────────────────────────────────────────────
    # AVALIAÇÃO
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(self, env) -> dict:
        """
        Avalia o agente treinado (ε=0, sem atualização da Q-table).

        Executa uma passagem completa pelo ambiente de teste usando
        apenas exploitation (ação greedy).

        Parameters
        ----------
        env : PortfolioEnv
            Ambiente de RL com dados de teste.

        Returns
        -------
        dict
            Métricas de avaliação:
            - 'portfolio_values': valor do portfólio a cada step
            - 'weights_history': pesos do portfólio a cada step
            - 'rewards': recompensa a cada step
            - 'actions': ações tomadas a cada step
            - 'returns': retorno do portfólio a cada step
            - 'final_value': valor final do portfólio
            - 'total_return': retorno acumulado percentual
            - 'sharpe_ratio': Sharpe Ratio anualizado
            - 'max_drawdown': drawdown máximo observado
        """
        state_continuous = env.reset()
        state = self.discretize(state_continuous)

        results = {
            "portfolio_values": [env.portfolio_value],
            "weights_history": [env.weights.copy()],
            "rewards": [],
            "actions": [],
            "returns": [],
        }

        done = False
        max_dd = 0.0

        while not done:
            # Ação greedy (sem exploração)
            q_values = self.q_table[state]
            action = int(np.argmax(q_values))

            # Executa no ambiente
            next_state_continuous, reward, done, info = env.step(action)
            next_state = self.discretize(next_state_continuous)

            # Registra
            results["portfolio_values"].append(info["portfolio_value"])
            results["weights_history"].append(info["weights"].copy())
            results["rewards"].append(reward)
            results["actions"].append(action)
            results["returns"].append(info["portfolio_return"])
            max_dd = max(max_dd, info["drawdown"])

            state = next_state

        # Métricas finais
        results["final_value"] = results["portfolio_values"][-1]
        results["total_return"] = (
            (results["final_value"] - env.initial_balance) / env.initial_balance
        )
        results["max_drawdown"] = max_dd

        # Sharpe Ratio anualizado
        daily_returns = np.array(results["returns"])
        if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
            sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe = 0.0
        results["sharpe_ratio"] = sharpe

        print(f"\n{'─'*50}")
        print(f"  AVALIAÇÃO NO TESTE (SARSA)")
        print(f"  Valor final:       R$ {results['final_value']:,.2f}")
        print(f"  Retorno acumulado: {results['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio:      {sharpe:.4f}")
        print(f"  Max Drawdown:      {max_dd*100:.2f}%")
        print(f"{'─'*50}\n")

        return results

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTÊNCIA
    # ─────────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """
        Salva o agente (Q-table, bins e hiperparâmetros) em arquivo.

        Parameters
        ----------
        filepath : str
            Caminho do arquivo para salvar (ex: 'resultados/td_agent.pkl').
        """
        data = {
            "q_table": dict(self.q_table),
            "bins": self.bins,
            "n_bins": self.n_bins,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "alpha_min": self.alpha_min,
            "alpha_decay": self.alpha_decay,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[SARSA] Agente salvo em: {filepath}")

    def load(self, filepath: str) -> None:
        """
        Carrega o agente a partir de um arquivo salvo.

        Parameters
        ----------
        filepath : str
            Caminho do arquivo para carregar.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions), data["q_table"]
        )
        self.bins = data["bins"]
        self.n_bins = data["n_bins"]
        self.n_actions = data["n_actions"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.alpha_min = data["alpha_min"]
        self.alpha_decay = data["alpha_decay"]
        self.epsilon_min = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]

        print(f"[SARSA] Agente carregado de: {filepath}")
        print(f"  Q-states: {len(self.q_table):,} | "
              f"ε: {self.epsilon:.4f} | α: {self.alpha:.4f}")