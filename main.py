
import os
import json
import numpy as np

from helpers.data_loader import load_train_data, load_test_data
from helpers.benchmarks import compute_benchmarks
from ambiente.portfolio_env import PortfolioEnv
from agentes.Q_learning import AgentQLearning



# Hiperparâmetros do agente
AGENT_CONFIG = {
    "n_bins": 5,
    "alpha": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "n_sim_episodes": 10,
    "seed": 42,
}

# Hiperparâmetros do ambiente
ENV_CONFIG = {
    "initial_balance": 100_000.0,
    "transaction_cost": 0.001,
    "weight_delta": 0.10,
    "alpha_diversification": 0.001,
    "beta_concentration": 0.002,
    "concentration_threshold": 0.60,
}

N_EPISODES = 500
LOG_INTERVAL = 50
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "resultados")


def main():


    print(" RL-WALLET-B3 — Q-Learning")
    print("Gestão de Portfólio com Aprendizado por Reforço")


    # --- Fase 1: Carregar dados ---
    print(" Carregando dados...\n")
    train_df = load_train_data()
    test_df = load_test_data()

    # --- Criar ambiente de treino ---
    print("\nCriando ambiente de treino...\n")
    train_env = PortfolioEnv(train_df, **ENV_CONFIG)
    print(f"  Ambiente criado: {train_env.n_steps} steps, "
          f"{train_env.n_actions} ações, "
          f"{train_env.observation_space_size} features de estado")

    # --- Criar agente Q-Learning ---
    print("\nCriando agente Q-Learning...\n")
    agent = AgentQLearning(train_env, **AGENT_CONFIG)

    # ---Treinar ---
    print("\nTreinamento...\n")
    history = agent.train(
        train_env,
        n_episodes=N_EPISODES,
        log_interval=LOG_INTERVAL,
    )

    # --- Avaliar numa base teste ---
    print("\nAvaliação no conjunto de teste...\n")
    test_env = PortfolioEnv(test_df, **ENV_CONFIG)
    eval_results = agent.evaluate(test_env)

    # --- Benchmarks ---
    print(" Comparação com benchmarks...\n")
    benchmarks = compute_benchmarks(test_df, ENV_CONFIG["initial_balance"])

    print(f"  {'Estratégia':<20} {'Valor Final':>15} {'Retorno':>10}")
    print(f"  {'─'*45}")
    print(f"  {'Q-Learning':<20} "
          f"R$ {eval_results['final_value']:>12,.2f} "
          f"{eval_results['total_return']*100:>8.2f}%")
    print(f"  {'Buy & Hold (1/3)':<20} "
          f"R$ {benchmarks['buy_hold']['final_value']:>12,.2f} "
          f"{benchmarks['buy_hold']['total_return']*100:>8.2f}%")
    print(f"  {'100% CDI':<20} "
          f"R$ {benchmarks['cdi']['final_value']:>12,.2f} "
          f"{benchmarks['cdi']['total_return']*100:>8.2f}%")

    # --- Fase 7: Salvar resultados ---
    print(f"\nSalvando resultados...\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Salvar agente
    agent_path = os.path.join(RESULTS_DIR, "q_agent.pkl")
    agent.save(agent_path)

    # Salvar métricas em JSON (para análise no notebook)
    metrics = {
        "agent": {
            "final_value": eval_results["final_value"],
            "total_return": eval_results["total_return"],
            "sharpe_ratio": eval_results["sharpe_ratio"],
            "max_drawdown": eval_results["max_drawdown"],
        },
        "benchmarks": {
            name: {
                "final_value": data["final_value"],
                "total_return": data["total_return"],
            }
            for name, data in benchmarks.items()
        },
        "training": {
            "n_episodes": N_EPISODES,
            "n_states_visited": history["n_states_visited"],
            "final_epsilon": history["episode_epsilons"][-1],
            "final_alpha": history["episode_alphas"][-1],
            "avg_reward_last_50": float(
                np.mean(history["episode_rewards"][-50:])
            ),
        },
        "config": {
            "agent": AGENT_CONFIG,
            "env": {k: v for k, v in ENV_CONFIG.items()},
        },
    }

    metrics_path = os.path.join(RESULTS_DIR, "metrics_qlearning.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Métricas salvas em: {metrics_path}")

    # Salvar histórico de treino (para plotar no notebook)
    training_path = os.path.join(RESULTS_DIR, "training_history.json")
    training_data = {
        "episode_rewards": [float(x) for x in history["episode_rewards"]],
        "episode_mean_rewards": [
            float(x) for x in history["episode_mean_rewards"]
        ],
        "episode_discounted_rewards": [
            float(x) for x in history["episode_discounted_rewards"]
        ],
        "episode_portfolio_values": [
            float(x) for x in history["episode_portfolio_values"]
        ],
        "episode_epsilons": [float(x) for x in history["episode_epsilons"]],
        "episode_alphas": [float(x) for x in history["episode_alphas"]],
        "episode_td_errors": [float(x) for x in history["episode_td_errors"]],
    }
    with open(training_path, "w") as f:
        json.dump(training_data, f, indent=2)
    print(f"  Histórico de treino salvo em: {training_path}")

    # Salvar valores do portfólio no teste (para plotar no notebook)
    eval_path = os.path.join(RESULTS_DIR, "eval_portfolio.json")
    eval_data = {
        "portfolio_values": [float(x) for x in eval_results["portfolio_values"]],
        "weights_history": [
            [float(w) for w in ws] for ws in eval_results["weights_history"]
        ],
        "actions": [int(a) for a in eval_results["actions"]],
        "returns": [float(r) for r in eval_results["returns"]],
        "benchmark_buy_hold": [float(x) for x in benchmarks["buy_hold"]["values"]],
        "benchmark_cdi": [float(x) for x in benchmarks["cdi"]["values"]],
    }
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"  Dados de avaliação salvos em: {eval_path}")

    print("="*60)


if __name__ == "__main__":
    main()
