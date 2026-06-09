"""
BaseAgent - funcionalidades compartilhadas por agentes tabulares.
"""

import pickle
from collections import defaultdict

import numpy as np

from helpers.feature_engineering import compute_bins_from_simulation, discretize_state


class BaseAgent:
    """Base comum para agentes Q-table com politica epsilon-greedy."""

    agent_name = "BaseAgent"

    def __init__(
        self,
        env,
        n_bins: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        n_sim_episodes: int = 10, # numero de episodios aleatorios para calcular os intervalo de discretizacao
        seed: int = 42,
    ):
        self.n_actions = env.n_actions # numero de acoes discretas possiveis (27 no caso do PortfolioEnv)
        self.n_bins = n_bins   # numero de bins para discretizacao de cada feature do estado
        self.alpha = alpha # taxa de aprendizado (learning rate)
        self.gamma = gamma # fator de desconto
        self.epsilon = epsilon # taxa de exploracao inicial
        self.epsilon_min = epsilon_min # taxa de exploracao minima
        self.epsilon_decay = epsilon_decay # fator de decaimento da taxa de exploracao
        self.seed = seed    
        self.rng = np.random.RandomState(seed)

        print(
            f"[{self.agent_name}] Calculando bins de discretizacao "
            f"({n_sim_episodes} episodios aleatorios)..."
        )
        self.bins = compute_bins_from_simulation(
            env, n_bins=n_bins, n_episodes=n_sim_episodes, seed=seed
        )
        print(f"[{self.agent_name}] Bins calculados para {len(self.bins)} features.")

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

    def discretize(self, state: np.ndarray) -> tuple:
        """Converte estado continuo em tupla discreta para a Q-table."""
        return discretize_state(state, self.bins)

    def choose_action(self, state: tuple) -> int:
        """Seleciona acao via politica epsilon-gulosa."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.randint(0, self.n_actions))

        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(self.rng.choice(best_actions))

    def decay_epsilon(self) -> None:
        """Decai epsilon apos cada episodio"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def evaluate(self, env) -> dict:
        """Avalia o agente treinado"""
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
            q_values = self.q_table[state]
            action = int(np.argmax(q_values))

            next_state_continuous, reward, done, info = env.step(action)
            next_state = self.discretize(next_state_continuous)

            results["portfolio_values"].append(info["portfolio_value"])
            results["weights_history"].append(info["weights"].copy())
            results["rewards"].append(reward)
            results["actions"].append(action)
            results["returns"].append(info["portfolio_return"])
            max_dd = max(max_dd, info["drawdown"])

            state = next_state

        results["final_value"] = results["portfolio_values"][-1]
        results["total_return"] = (
            (results["final_value"] - env.initial_balance) / env.initial_balance
        )
        results["max_drawdown"] = max_dd

        daily_returns = np.array(results["returns"])
        if len(daily_returns) > 1 and np.std(daily_returns) > 1e-10:
            sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
        else:
            sharpe = 0.0
        results["sharpe_ratio"] = sharpe

        print(f"\n{'-'*50}")
        print(f"  AVALIACAO NO TESTE ({self.agent_name})")
        print(f"  Valor final:       R$ {results['final_value']:,.2f}")
        print(f"  Retorno acumulado: {results['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio:      {sharpe:.4f}")
        print(f"  Max Drawdown:      {max_dd*100:.2f}%")
        print(f"{'-'*50}\n")

        return results

    def save(self, filepath: str) -> None:
        """Salva Q-table, bins e hiperparametros em arquivo pickle."""
        data = {
            "q_table": dict(self.q_table),
            "bins": self.bins,
            "n_bins": self.n_bins,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[{self.agent_name}] Agente salvo em: {filepath}")

    def load(self, filepath: str) -> None:
        """Carrega Q-table, bins e hiperparametros de arquivo pickle."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.n_bins = data["n_bins"]
        self.n_actions = data["n_actions"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.epsilon_min = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        self.bins = data["bins"]
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions), data["q_table"]
        )

        print(f"[{self.agent_name}] Agente carregado de: {filepath}")
        print(
            f"  Q-states: {len(self.q_table):,} | "
            f"epsilon: {self.epsilon:.4f} | alpha: {self.alpha:.4f}"
        )
