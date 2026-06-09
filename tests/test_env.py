"""
Smoke tests para o PortfolioEnv.

Valida instanciação, reset, step, loop completo, normalização de pesos
e custo zero ao manter posição.

Executar:
    python tests/test_env.py
    # ou
    python -m pytest tests/test_env.py -v
"""

import sys
import os
import numpy as np
import pandas as pd

# Garante que o pacote raiz está no path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ambiente.portfolio_env import PortfolioEnv


DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dados", "treino.csv"
)


def _load_env():
    """Helper: carrega dados e cria ambiente."""
    df = pd.read_csv(DATA_PATH)
    # Preencher NaN nos retornos (primeira linha) com 0
    df = df.fillna(0.0)
    return PortfolioEnv(df)


def test_instanciacao():
    """1. Instanciação sem crash."""
    env = _load_env()
    assert env.n_assets == 3
    assert env.n_actions == 27
    assert env.observation_space_size == 6
    print("[OK] test_instanciacao")


def test_reset():
    """2. Reset retorna array shape (6,) com valores finitos."""
    env = _load_env()
    obs = env.reset()
    assert obs.shape == (6,), f"Shape esperado (6,), recebido {obs.shape}"
    assert np.all(np.isfinite(obs)), "Observação contém NaN ou Inf"
    print("[OK] test_reset")


def test_step_manter():
    """3. Step com ação 'manter' (13) retorna tupla correta."""
    env = _load_env()
    obs = env.reset()

    # Ação 13 = (MANTER, MANTER, MANTER) → deltas = [0, 0, 0]
    obs2, reward, done, info = env.step(13)

    assert obs2.shape == (6,), f"Shape esperado (6,), recebido {obs2.shape}"
    assert np.all(np.isfinite(obs2)), "Observação contém NaN ou Inf"
    assert isinstance(reward, (float, np.floating)), "Reward deve ser float"
    assert np.isfinite(reward), "Reward deve ser finito"
    assert isinstance(done, bool), "Done deve ser bool"
    assert "portfolio_value" in info
    assert "weights" in info
    print("[OK] test_step_manter")


def test_loop_completo():
    """4. Loop com ações aleatórias até done=True, portfolio_value > 0."""
    env = _load_env()
    obs = env.reset()
    done = False
    steps = 0

    rng = np.random.default_rng(42)

    while not done:
        action = rng.integers(0, env.n_actions)
        obs, reward, done, info = env.step(action)
        steps += 1

    assert steps > 0, "Nenhum step executado"
    assert info["portfolio_value"] > 0, "Portfolio value deve ser positivo"
    print(f"[OK] test_loop_completo ({steps} steps, "
          f"valor final: R$ {info['portfolio_value']:,.2f})")


def test_pesos_normalizados():
    """5. A cada step, sum(weights) ≈ 1.0."""
    env = _load_env()
    env.reset()

    rng = np.random.default_rng(123)

    for _ in range(50):
        action = rng.integers(0, env.n_actions)
        _, _, done, info = env.step(action)
        weight_sum = np.sum(info["weights"])
        assert abs(weight_sum - 1.0) < 1e-8, (
            f"Pesos não normalizados: sum={weight_sum}"
        )
        if done:
            break

    print("[OK] test_pesos_normalizados")


def test_custo_zero_manter():
    """6. Ação 'manter' (13) deve ter custo ≈ 0."""
    env = _load_env()
    env.reset()

    _, _, _, info = env.step(13)

    assert info["cost"] < 1e-10, (
        f"Custo deveria ser ~0 ao manter, recebido {info['cost']}"
    )
    print("[OK] test_custo_zero_manter")


if __name__ == "__main__":
    print("=" * 60)
    print("  Smoke Tests — PortfolioEnv")
    print("=" * 60)
    print()

    test_instanciacao()
    test_reset()
    test_step_manter()
    test_loop_completo()
    test_pesos_normalizados()
    test_custo_zero_manter()

    print()
    print("=" * 60)
    print("  TODOS OS TESTES PASSARAM ✓")
    print("=" * 60)
