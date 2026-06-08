import os
import json
import itertools
import numpy as np

from src.data_loader import load_train_data, load_test_data
from ambiente.portfolio_env import PortfolioEnv
from agentes.Q_learning import AgentQLearning

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "resultados")

BASE_AGENT_CONFIG = {
    "n_bins": 5,
    "alpha": 0.1,
    "alpha_min": 0.01,
    "alpha_decay": 0.995,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "n_sim_episodes": 2,
    "seed": 42,
}

BASE_ENV_CONFIG = {
    "initial_balance": 100_000.0,
    "transaction_cost": 0.001,
    "window_size": 20,
    "weight_delta": 0.10,
    "lambda_transaction": 0.1,
    "alpha_benchmark": 0.5,
    "alpha_diversification": 0.2,
    "beta_drawdown": 2.0,
    "beta_concentration": 1.0,
    "drawdown_threshold": 0.10,
    "concentration_threshold": 0.60,
}

N_EPISODES = 5

def run_experiment(train_df, test_df, env_config, agent_config, run_name):
    print(f"\n--- Iniciando experimento: {run_name} ---")
    train_env = PortfolioEnv(train_df, **env_config)
    agent = AgentQLearning(train_env, **agent_config)
    
    # Reduzindo LOG_INTERVAL para evitar muito print
    history = agent.train(train_env, n_episodes=N_EPISODES, log_interval=N_EPISODES)
    
    test_env = PortfolioEnv(test_df, **env_config)
    eval_results = agent.evaluate(test_env)
    
    metrics = {
        "run_name": run_name,
        "config": {
            "agent": agent_config,
            "env": env_config
        },
        "results": {
            "final_value": eval_results["final_value"],
            "total_return": eval_results["total_return"],
            "sharpe_ratio": eval_results["sharpe_ratio"],
            "max_drawdown": eval_results["max_drawdown"],
        },
        "training": {
            "n_states_visited": history["n_states_visited"],
            "avg_reward_last_50": float(np.mean(history["episode_rewards"][-50:])),
        }
    }
    
    print(f"Resultado {run_name}: Retorno {eval_results['total_return']*100:.2f}%, Sharpe {eval_results['sharpe_ratio']:.2f}, Drawdown {eval_results['max_drawdown']:.2f}")
    return metrics

def main():
    print("Carregando dados...")
    train_df = load_train_data()
    test_df = load_test_data()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    
    # 1. Variando n_bins
    print("\n\n=== EXPERIMENTO 1: Variação de n_bins ===")
    bins_values = [3, 5, 8, 10]
    for n in bins_values:
        agent_config = BASE_AGENT_CONFIG.copy()
        agent_config["n_bins"] = n
        run_name = f"bins_{n}"
        res = run_experiment(train_df, test_df, BASE_ENV_CONFIG, agent_config, run_name)
        all_results.append(res)
        
    # 2. Variando epsilon_decay e alpha_decay
    print("\n\n=== EXPERIMENTO 2: Variação de Taxas de Decaimento (epsilon, alpha) ===")
    decays = [0.99, 0.995, 0.999]
    for eps_dec, alp_dec in itertools.product(decays, decays):
        agent_config = BASE_AGENT_CONFIG.copy()
        agent_config["epsilon_decay"] = eps_dec
        agent_config["alpha_decay"] = alp_dec
        run_name = f"epsDecay_{eps_dec}_alpDecay_{alp_dec}"
        res = run_experiment(train_df, test_df, BASE_ENV_CONFIG, agent_config, run_name)
        all_results.append(res)
        
    # 3. Variando transaction_cost e weight_delta
    print("\n\n=== EXPERIMENTO 3: Variação de Ambiente (transaction_cost, weight_delta) ===")
    costs = [0.000, 0.001, 0.005]
    deltas = [0.05, 0.10, 0.20]
    for cost, delta in itertools.product(costs, deltas):
        env_config = BASE_ENV_CONFIG.copy()
        env_config["transaction_cost"] = cost
        env_config["weight_delta"] = delta
        run_name = f"txCost_{cost}_wDelta_{delta}"
        
        # Para testar adequadamente o impacto no agente, precisamos treiná-lo e avaliá-lo no novo ambiente
        res = run_experiment(train_df, test_df, env_config, BASE_AGENT_CONFIG, run_name)
        all_results.append(res)
        
    # Salvar resultados
    output_file = os.path.join(RESULTS_DIR, "experiments_q_learning_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"\n\nTodos os experimentos concluídos! Resultados salvos em: {output_file}")

if __name__ == "__main__":
    main()
