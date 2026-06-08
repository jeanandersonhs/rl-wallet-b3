"""
feature_engineering.py — Discretização de estados para Q-Learning tabular.

Responsável por:
  - Calcular limites dos bins a partir de simulação com ações aleatórias
  - Discretizar vetor de estado contínuo (18D) em tupla inteira

O PortfolioEnv retorna estados contínuos (18 dimensões). Para o Q-Learning
tabular, cada feature é mapeada para um de K bins usando np.digitize.

Referência: documentacao.txt, seções 3.1 e 4.1.
"""

import numpy as np


def compute_bins_from_simulation(
    env,
    n_bins: int = 5,
    n_episodes: int = 10,
    seed: int = 42,
) -> list[np.ndarray]:
    """
    Calcula os limites dos bins executando episódios com ações aleatórias.

    Simula o ambiente com política aleatória para capturar a distribuição
    real dos estados (incluindo features dinâmicas como drawdown, pesos e
    volatilidade do portfólio).

    Parameters
    ----------
    env : PortfolioEnv
        Ambiente de RL inicializado com dados de treino.
    n_bins : int
        Número de bins por feature (default: 5).
    n_episodes : int
        Número de episódios de simulação aleatória (default: 10).
    seed : int
        Seed para reprodutibilidade.

    Returns
    -------
    list[np.ndarray]
        Lista com 18 arrays, cada um contendo (n_bins - 1) limites de corte.
    """
    rng = np.random.RandomState(seed)
    all_states = []

    for _ in range(n_episodes):
        state = env.reset()
        all_states.append(state.copy())
        done = False

        while not done:
            action = rng.randint(0, env.n_actions)
            state, _, done, _ = env.step(action)
            all_states.append(state.copy())

    states_matrix = np.array(all_states)  # shape (N, 18)
    n_features = states_matrix.shape[1]

    bins = []
    for i in range(n_features):
        # Percentis uniformemente espaçados para bins equilibrados
        percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        bin_edges = np.percentile(states_matrix[:, i], percentiles)

        # Remover duplicatas (pode ocorrer se uma feature é constante)
        bin_edges = np.unique(bin_edges)
        bins.append(bin_edges)

    return bins


def discretize_state(state: np.ndarray, bins: list[np.ndarray]) -> tuple:
    """
    Converte um vetor de estado contínuo em tupla discreta.

    Parameters
    ----------
    state : np.ndarray
        Vetor de estado contínuo, shape (18,).
    bins : list[np.ndarray]
        Limites dos bins calculados por compute_bins_from_simulation().

    Returns
    -------
    tuple[int, ...]
        Tupla de inteiros representando o estado discreto.
        Cada valor está no intervalo [0, len(bins[i])].
    """
    discrete = []
    for i, val in enumerate(state):
        discrete.append(int(np.digitize(val, bins[i])))
    return tuple(discrete)
