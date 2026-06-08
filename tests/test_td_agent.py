"""
test_td_agent.py — Smoke tests para o agente SARSA (AgentTD).

Valida:
    - Criação do agente com bins e Q-table
    - Discretização de estados
    - Política ε-greedy
    - Atualização SARSA (on-policy: usa Q(s',a'), não max)
    - Treinamento de 1 episódio
    - Avaliação completa
    - Persistência (save/load)

Referência: documentacao.txt, seção 4.2.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ambiente.portfolio_env import PortfolioEnv
from agentes.TD_learning import AgentTD


# ─────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """
    Cria um DataFrame sintético com 60 dias de dados (> window_size=20).
    Estrutura idêntica ao treino.csv real.
    """
    np.random.seed(42)
    n = 60

    prices_itub4 = 30.0 + np.cumsum(np.random.randn(n) * 0.3)
    prices_bova11 = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    prices_bbas3 = 25.0 + np.cumsum(np.random.randn(n) * 0.4)

    ret_itub4 = np.concatenate([[0.0], np.diff(prices_itub4) / prices_itub4[:-1]])
    ret_bova11 = np.concatenate([[0.0], np.diff(prices_bova11) / prices_bova11[:-1]])
    ret_bbas3 = np.concatenate([[0.0], np.diff(prices_bbas3) / prices_bbas3[:-1]])

    cdi = np.full(n, 0.0004)  # ~10% a.a.

    df = pd.DataFrame({
        "data": pd.date_range("2023-01-02", periods=n, freq="B"),
        "ITUB4": prices_itub4,
        "BOVA11": prices_bova11,
        "BBAS3": prices_bbas3,
        "retorno_ITUB4": ret_itub4,
        "retorno_BOVA11": ret_bova11,
        "retorno_BBAS3": ret_bbas3,
        "CDI_decimal": cdi,
        "retorno_acima_cdi_ITUB4": ret_itub4 - cdi,
        "retorno_acima_cdi_BOVA11": ret_bova11 - cdi,
        "retorno_acima_cdi_BBAS3": ret_bbas3 - cdi,
    })
    return df


@pytest.fixture
def env(sample_df):
    """Cria um PortfolioEnv com dados sintéticos."""
    return PortfolioEnv(sample_df, window_size=20)


@pytest.fixture
def agent(env):
    """Cria um AgentTD com configuração padrão."""
    return AgentTD(
        env,
        n_bins=5,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        n_sim_episodes=2,
        seed=42,
    )


# ─────────────────────────────────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────────────────────────────────

class TestAgentCreation:
    """Testes de instanciação do agente."""

    def test_agent_creation(self, agent):
        """Agente criado com bins, Q-table vazia e hiperparâmetros corretos."""
        assert agent.n_actions == 27
        assert agent.n_bins == 5
        assert agent.alpha == 0.1
        assert agent.gamma == 0.99
        assert agent.epsilon == 1.0
        assert len(agent.bins) == 18  # 18 features do estado
        assert len(agent.q_table) == 0  # Q-table começa vazia

    def test_bins_computed(self, agent):
        """Bins calculados para cada feature do estado."""
        for i, bin_edges in enumerate(agent.bins):
            assert isinstance(bin_edges, np.ndarray)
            # Cada feature deve ter pelo menos 1 bin edge
            assert len(bin_edges) >= 1, f"Feature {i} sem bin edges"


class TestDiscretization:
    """Testes de discretização de estados."""

    def test_discretize_returns_tuple(self, agent, env):
        """Discretização retorna tupla de inteiros."""
        state = env.reset()
        discrete = agent.discretize(state)
        assert isinstance(discrete, tuple)
        assert len(discrete) == 18  # 18 features

    def test_discretize_values_are_ints(self, agent, env):
        """Cada elemento da tupla discreta é um inteiro."""
        state = env.reset()
        discrete = agent.discretize(state)
        for val in discrete:
            assert isinstance(val, (int, np.integer))


class TestEpsilonGreedy:
    """Testes da política ε-greedy."""

    def test_choose_action_valid_range(self, agent, env):
        """Ação selecionada está no intervalo [0, 26]."""
        state = env.reset()
        discrete = agent.discretize(state)
        for _ in range(50):
            action = agent.choose_action(discrete)
            assert 0 <= action < 27

    def test_exploration_with_high_epsilon(self, agent, env):
        """Com ε=1.0, todas as ações devem ser aleatórias (distribuição)."""
        agent.epsilon = 1.0
        state = env.reset()
        discrete = agent.discretize(state)
        actions = [agent.choose_action(discrete) for _ in range(1000)]
        unique_actions = set(actions)
        # Com 1000 tentativas aleatórias, esperamos ver a maioria das 27 ações
        assert len(unique_actions) > 15

    def test_exploitation_with_zero_epsilon(self, agent, env):
        """Com ε=0, deve sempre retornar a ação greedy."""
        agent.epsilon = 0.0
        state = env.reset()
        discrete = agent.discretize(state)
        # Define Q-values distintos para ter uma ação greedy clara
        agent.q_table[discrete] = np.arange(27, dtype=np.float64)
        actions = [agent.choose_action(discrete) for _ in range(100)]
        # Deve sempre retornar ação 26 (maior Q-value)
        assert all(a == 26 for a in actions)


class TestSarsaUpdate:
    """Testes da atualização SARSA (diferença fundamental vs Q-Learning)."""

    def test_update_changes_q_value(self, agent, env):
        """Q-value muda após atualização SARSA."""
        state = env.reset()
        s = agent.discretize(state)
        next_state, _, _, _ = env.step(0)
        s_next = agent.discretize(next_state)

        q_before = agent.q_table[s][0]
        agent.update(s, 0, 1.0, s_next, 0, False)
        q_after = agent.q_table[s][0]

        assert q_after != q_before

    def test_sarsa_uses_next_action_not_max(self, agent, env):
        """SARSA usa Q(s', a') e NÃO max Q(s', a')."""
        state = env.reset()
        s = agent.discretize(state)
        next_state, _, _, _ = env.step(0)
        s_next = agent.discretize(next_state)

        # Configura Q-values diferentes para s'
        agent.q_table[s_next] = np.zeros(27)
        agent.q_table[s_next][5] = 10.0   # melhor ação
        agent.q_table[s_next][2] = 3.0    # ação que vamos usar (a')

        # Reset Q(s, 0) para zero
        agent.q_table[s][0] = 0.0

        # Update com next_action=2 (NÃO a melhor ação 5)
        agent.update(s, 0, 1.0, s_next, 2, False)

        # Q(s,0) deveria usar Q(s', 2) = 3.0, não Q(s', 5) = 10.0
        # TD target = r + γ * Q(s', a') = 1.0 + 0.99 * 3.0 = 3.97
        # Q(s,0) = 0 + α * (3.97 - 0) = 0.1 * 3.97 = 0.397
        expected = 0.1 * (1.0 + 0.99 * 3.0)
        assert abs(agent.q_table[s][0] - expected) < 1e-10

        # Se fosse Q-Learning (max), seria:
        # TD target = 1.0 + 0.99 * 10.0 = 10.9
        # Q(s,0) = 0.1 * 10.9 = 1.09  (diferente!)
        qlearning_value = 0.1 * (1.0 + 0.99 * 10.0)
        assert abs(agent.q_table[s][0] - qlearning_value) > 0.5

    def test_update_terminal_state(self, agent, env):
        """Update em estado terminal: td_target = r (sem futuro)."""
        state = env.reset()
        s = agent.discretize(state)
        next_state, _, _, _ = env.step(0)
        s_next = agent.discretize(next_state)

        agent.q_table[s][0] = 0.0
        agent.update(s, 0, 5.0, s_next, 0, done=True)

        # td_target = reward = 5.0 (ignora próximo estado)
        # Q(s,0) = 0 + 0.1 * (5.0 - 0) = 0.5
        assert abs(agent.q_table[s][0] - 0.5) < 1e-10

    def test_td_error_returned(self, agent, env):
        """Update retorna TD error."""
        state = env.reset()
        s = agent.discretize(state)
        next_state, _, _, _ = env.step(0)
        s_next = agent.discretize(next_state)

        td_error = agent.update(s, 0, 1.0, s_next, 0, False)
        assert isinstance(td_error, float)


class TestDecay:
    """Testes de decay de hiperparâmetros."""

    def test_epsilon_decay(self, agent):
        """ε decai a cada chamada."""
        eps_before = agent.epsilon
        agent.decay_hyperparams()
        assert agent.epsilon < eps_before
        assert agent.epsilon >= agent.epsilon_min

    def test_alpha_decay(self, agent):
        """α decai a cada chamada."""
        alpha_before = agent.alpha
        agent.decay_hyperparams()
        assert agent.alpha < alpha_before
        assert agent.alpha >= agent.alpha_min


class TestTraining:
    """Testes de treinamento."""

    def test_train_1_episode(self, agent, env):
        """Treina 1 episódio sem erros e retorna histórico."""
        history = agent.train(env, n_episodes=1, log_interval=1)

        assert "episode_rewards" in history
        assert "episode_portfolio_values" in history
        assert "episode_epsilons" in history
        assert "episode_alphas" in history
        assert "episode_td_errors" in history
        assert "n_states_visited" in history

        assert len(history["episode_rewards"]) == 1
        assert len(history["episode_portfolio_values"]) == 1

    def test_train_multiple_episodes(self, agent, env):
        """Treina 3 episódios, histórico tem tamanho correto."""
        history = agent.train(env, n_episodes=3, log_interval=1)
        assert len(history["episode_rewards"]) == 3
        assert history["n_states_visited"] > 0


class TestEvaluation:
    """Testes de avaliação."""

    def test_evaluate_returns_metrics(self, agent, env):
        """Avaliação retorna todas as métricas esperadas."""
        # Treinar minimamente antes de avaliar
        agent.train(env, n_episodes=1, log_interval=1)

        results = agent.evaluate(env)

        assert "portfolio_values" in results
        assert "weights_history" in results
        assert "rewards" in results
        assert "actions" in results
        assert "returns" in results
        assert "final_value" in results
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results

    def test_evaluate_portfolio_values_length(self, agent, env):
        """Número de portfolio values = steps executados + 1 (inicial)."""
        agent.train(env, n_episodes=1, log_interval=1)
        results = agent.evaluate(env)

        # Deve ter pelo menos 2 valores (inicial + pelo menos 1 step)
        assert len(results["portfolio_values"]) >= 2

    def test_evaluate_actions_valid(self, agent, env):
        """Ações na avaliação estão no intervalo [0, 26]."""
        agent.train(env, n_episodes=1, log_interval=1)
        results = agent.evaluate(env)

        for action in results["actions"]:
            assert 0 <= action < 27


class TestPersistence:
    """Testes de save/load."""

    def test_save_load(self, agent, env):
        """Agente salvo e carregado mantém Q-table e hiperparâmetros."""
        # Treinar para popular a Q-table
        agent.train(env, n_episodes=2, log_interval=1)
        n_states = len(agent.q_table)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            agent.save(filepath)
            assert os.path.exists(filepath)

            # Criar novo agente e carregar
            agent2 = AgentTD(env, n_sim_episodes=1, seed=42)
            agent2.load(filepath)

            assert len(agent2.q_table) == n_states
            assert agent2.n_bins == agent.n_bins
            assert agent2.gamma == agent.gamma
        finally:
            os.unlink(filepath)
