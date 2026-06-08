"""
metricas.py — Comparação Q-Learning vs TD_LEARNING com gráficos.

Métricas:
  1. Recompensa Acumulada por Episódio (Reward Curve)
  2. Convergência da Função de Valor (Q-Value / TD Error)
  3. Eficiência de Amostra (Sample Efficiency)
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Adiciona raiz do projeto ao path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "resultados")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figuras")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Configuração visual ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COLORS = {"qlearning": "#2196F3", "TD_LEARNING": "#FF5722"}


def load_history(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return json.load(f)


def smooth(values, window=20):
    """Média móvel simples para suavizar curvas."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ═════════════════════════════════════════════════════════════════════
# 1. RECOMPENSA ACUMULADA POR EPISÓDIO (Reward Curve)
# ═════════════════════════════════════════════════════════════════════

def plot_reward_curve():
    """
    Métrica 'rei' do RL — soma total de recompensas por episódio.
    Equivalente à Acurácia em aprendizado supervisionado.
    """
    q_hist = load_history("training_history.json")
    td_hist = load_history("training_history_td.json")

    q_rewards = q_hist["episode_rewards"]
    td_rewards = td_hist["episode_rewards"]
    episodes = range(1, len(q_rewards) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Painel 1: Curvas brutas + suavizadas ---
    ax = axes[0]
    ax.plot(episodes, q_rewards, alpha=0.15, color=COLORS["qlearning"])
    ax.plot(episodes, td_rewards, alpha=0.15, color=COLORS["TD_LEARNING"])

    w = 20
    q_smooth = smooth(q_rewards, w)
    td_smooth = smooth(td_rewards, w)
    ep_smooth = range(w, len(q_rewards) + 1)

    ax.plot(ep_smooth, q_smooth, color=COLORS["qlearning"],
            linewidth=2, label=f"Q-Learning (média {w} ep)")
    ax.plot(ep_smooth, td_smooth, color=COLORS["TD_LEARNING"],
            linewidth=2, label=f"TD_LEARNING (média {w} ep)")

    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa Total do Episódio")
    ax.set_title("1. Recompensa Acumulada por Episódio")
    ax.legend(loc="lower right")

    # --- Painel 2: Valor do portfólio por episódio ---
    ax2 = axes[1]
    q_values = q_hist["episode_portfolio_values"]
    td_values = td_hist["episode_portfolio_values"]

    ax2.plot(episodes, q_values, alpha=0.15, color=COLORS["qlearning"])
    ax2.plot(episodes, td_values, alpha=0.15, color=COLORS["TD_LEARNING"])

    q_val_smooth = smooth(q_values, w)
    td_val_smooth = smooth(td_values, w)

    ax2.plot(ep_smooth, q_val_smooth, color=COLORS["qlearning"],
             linewidth=2, label=f"Q-Learning")
    ax2.plot(ep_smooth, td_val_smooth, color=COLORS["TD_LEARNING"],
             linewidth=2, label=f"TD_LEARNING")

    ax2.axhline(y=100_000, color="gray", linestyle="--",
                alpha=0.5, label="Capital Inicial (R$100k)")
    ax2.set_xlabel("Episódio")
    ax2.set_ylabel("Valor Final do Portfólio (R$)")
    ax2.set_title("Valor do Portfólio ao Final de Cada Episódio")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_reward_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Salvo em: {path}")


# ═════════════════════════════════════════════════════════════════════
# 2. CONVERGÊNCIA DA FUNÇÃO DE VALOR (TD Error)
# ═════════════════════════════════════════════════════════════════════

def plot_convergence():
    """
    Monitora o TD Error médio por episódio.
    TD Error → 0 indica que as estimativas Q(s,a) estão convergindo.
    """
    q_hist = load_history("training_history.json")
    td_hist = load_history("training_history_td.json")

    q_td_errors = q_hist["episode_td_errors"]
    td_td_errors = td_hist["episode_td_errors"]
    episodes = range(1, len(q_td_errors) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Painel 1: TD Error ---
    ax = axes[0]
    w = 20
    q_smooth = smooth(q_td_errors, w)
    td_smooth_vals = smooth(td_td_errors, w)
    ep_smooth = range(w, len(q_td_errors) + 1)

    ax.plot(episodes, q_td_errors, alpha=0.12, color=COLORS["qlearning"])
    ax.plot(episodes, td_td_errors, alpha=0.12, color=COLORS["TD_LEARNING"])
    ax.plot(ep_smooth, q_smooth, color=COLORS["qlearning"],
            linewidth=2, label="Q-Learning")
    ax.plot(ep_smooth, td_smooth_vals, color=COLORS["TD_LEARNING"],
            linewidth=2, label="TD_LEARNING")

    ax.set_xlabel("Episódio")
    ax.set_ylabel("TD Error Médio (|δ|)")
    ax.set_title("2. Convergência — TD Error por Episódio")
    ax.legend()

    # --- Painel 2: Epsilon e Alpha decay ---
    ax2 = axes[1]
    q_eps = q_hist["episode_epsilons"]
    td_eps = td_hist["episode_epsilons"]
    q_alpha = q_hist["episode_alphas"]

    ax2.plot(episodes, q_eps, color=COLORS["qlearning"],
             linewidth=2, label="ε (exploração)")
    ax2.plot(episodes, q_alpha, color=COLORS["qlearning"],
             linewidth=2, linestyle="--", label="α (learning rate)")

    ax2.set_xlabel("Episódio")
    ax2.set_ylabel("Valor do Hiperparâmetro")
    ax2.set_title("Decay de ε e α ao Longo do Treinamento")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Salvo em: {path}")


# ═════════════════════════════════════════════════════════════════════
# 3. EFICIÊNCIA DE AMOSTRA (Sample Efficiency)
# ═════════════════════════════════════════════════════════════════════

def plot_sample_efficiency():
    """
    Compara quantos episódios cada algoritmo precisou para se tornar
    consistentemente lucrativo (portfólio > capital inicial).
    """
    q_hist = load_history("training_history.json")
    td_hist = load_history("training_history_td.json")

    q_values = q_hist["episode_portfolio_values"]
    td_values = td_hist["episode_portfolio_values"]

    initial = 100_000.0
    w = 30  # janela para "consistentemente lucrativo"

    def find_profitable_episode(values, window):
        """Primeiro episódio onde a média móvel fica acima do capital."""
        smoothed = smooth(values, window)
        for i, v in enumerate(smoothed):
            if v > initial:
                return i + window
        return None

    q_profit_ep = find_profitable_episode(q_values, w)
    td_profit_ep = find_profitable_episode(td_values, w)

    # Retorno acumulado percentual por episódio
    q_returns = [(v - initial) / initial * 100 for v in q_values]
    td_returns = [(v - initial) / initial * 100 for v in td_values]
    episodes = range(1, len(q_returns) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Painel 1: Retorno % por episódio ---
    ax = axes[0]
    q_ret_smooth = smooth(q_returns, w)
    td_ret_smooth = smooth(td_returns, w)
    ep_smooth = range(w, len(q_returns) + 1)

    ax.plot(ep_smooth, q_ret_smooth, color=COLORS["qlearning"],
            linewidth=2, label="Q-Learning")
    ax.plot(ep_smooth, td_ret_smooth, color=COLORS["TD_LEARNING"],
            linewidth=2, label="TD_LEARNING")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5,
               label="Break-even (0%)")

    if q_profit_ep:
        ax.axvline(x=q_profit_ep, color=COLORS["qlearning"],
                   linestyle=":", alpha=0.7)
        ax.annotate(f"Q-Learning: ep {q_profit_ep}",
                    xy=(q_profit_ep, 0), fontsize=9,
                    color=COLORS["qlearning"],
                    xytext=(q_profit_ep + 20, -3))

    if td_profit_ep:
        ax.axvline(x=td_profit_ep, color=COLORS["TD_LEARNING"],
                   linestyle=":", alpha=0.7)
        ax.annotate(f"TD_LEARNING: ep {td_profit_ep}",
                    xy=(td_profit_ep, 0), fontsize=9,
                    color=COLORS["TD_LEARNING"],
                    xytext=(td_profit_ep + 20, 3))

    ax.set_xlabel("Episódio")
    ax.set_ylabel("Retorno (%)")
    ax.set_title("3. Eficiência de Amostra — Retorno por Episódio")
    ax.legend()

    # --- Painel 2: Tabela de métricas finais ---
    ax2 = axes[1]
    ax2.axis("off")

    q_metrics = json.load(open(os.path.join(RESULTS_DIR,
                                            "metrics_qlearning.json")))
    td_metrics = json.load(open(os.path.join(RESULTS_DIR,
                                             "metrics_td.json")))

    table_data = [
        ["Métrica", "Q-Learning", "TD_LEARNING"],
        ["Valor Final (R$)",
         f"{q_metrics['agent']['final_value']:,.2f}",
         f"{td_metrics['agent']['final_value']:,.2f}"],
        ["Retorno (%)",
         f"{q_metrics['agent']['total_return']*100:.2f}%",
         f"{td_metrics['agent']['total_return']*100:.2f}%"],
        ["Sharpe Ratio",
         f"{q_metrics['agent']['sharpe_ratio']:.4f}",
         f"{td_metrics['agent']['sharpe_ratio']:.4f}"],
        ["Max Drawdown",
         f"{q_metrics['agent']['max_drawdown']*100:.2f}%",
         f"{td_metrics['agent']['max_drawdown']*100:.2f}%"],
        ["Estados Visitados",
         f"{q_metrics['training']['n_states_visited']:,}",
         f"{td_metrics['training']['n_states_visited']:,}"],
        ["Ep. Lucrativo",
         str(q_profit_ep or "N/A"),
         str(td_profit_ep or "N/A")],
    ]

    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Colorir header
    for j in range(3):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(weight="bold")

    ax2.set_title("Comparação Final — Teste (2024)", fontsize=14,
                  pad=20)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_sample_efficiency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Salvo em: {path}")


# ═════════════════════════════════════════════════════════════════════
# 4. SENSIBILIDADE A HIPERPARÂMETROS (Learning Rate)
# ═════════════════════════════════════════════════════════════════════

def plot_hyperparameter_sensitivity():
    """
    Treina ambos os agentes com diferentes learning rates e compara
    o impacto no retorno final.
    """
    from helpers.data_loader import load_train_data
    from ambiente.portfolio_env import PortfolioEnv
    from agentes.Q_learning import AgentQLearning
    from agentes.TD_learning import AgentTD

    print("\n  Treinando com diferentes learning rates...")
    print("  (isso pode levar alguns minutos)\n")

    train_df = load_train_data()
    alphas = [0.01, 0.05, 0.1, 0.2, 0.5]
    n_episodes = 100  # reduzido para velocidade

    env_config = {
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

    q_results = {}
    td_results = {}

    for alpha in alphas:
        print(f"    α = {alpha}...")
        env = PortfolioEnv(train_df, **env_config)

        # Q-Learning
        q_agent = AgentQLearning(
            env, n_bins=5, alpha=alpha, alpha_min=0.005,
            alpha_decay=0.995, gamma=0.99, epsilon=1.0,
            epsilon_min=0.01, epsilon_decay=0.995,
            n_sim_episodes=3, seed=42,
        )
        q_hist = q_agent.train(env, n_episodes=n_episodes, log_interval=999)
        q_results[alpha] = q_hist["episode_portfolio_values"]

        # TD_LEARNING
        env2 = PortfolioEnv(train_df, **env_config)
        td_agent = AgentTD(
            env2, n_bins=5, alpha=alpha, alpha_min=0.005,
            alpha_decay=0.995, gamma=0.99, epsilon=1.0,
            epsilon_min=0.01, epsilon_decay=0.995,
            n_sim_episodes=3, seed=42,
        )
        td_hist = td_agent.train(env2, n_episodes=n_episodes, log_interval=999)
        td_results[alpha] = td_hist["episode_portfolio_values"]

    # --- Plotar ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(alphas) - 1)

    for i, alpha in enumerate(alphas):
        color = cmap(norm(i))
        w = 10
        episodes = range(1, n_episodes + 1)

        q_sm = smooth(q_results[alpha], w)
        td_sm = smooth(td_results[alpha], w)
        ep_sm = range(w, n_episodes + 1)

        axes[0].plot(ep_sm, q_sm, color=color, linewidth=2,
                     label=f"α={alpha}")
        axes[1].plot(ep_sm, td_sm, color=color, linewidth=2,
                     label=f"α={alpha}")

    for ax, title in zip(axes, ["Q-Learning", "TD_LEARNING"]):
        ax.axhline(y=100_000, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("Episódio")
        ax.set_ylabel("Valor do Portfólio (R$)")
        ax.set_title(f"4. Sensibilidade ao α — {title}")
        ax.legend(title="Learning Rate", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_alpha_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Salvo em: {path}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MÉTRICAS DE COMPARAÇÃO — Q-Learning vs TD_LEARNING")
    print("=" * 60)

    print("\n1. Recompensa Acumulada por Episódio...")
    plot_reward_curve()

    print("\n2. Convergência da Função de Valor...")
    plot_convergence()

    print("\n3. Eficiência de Amostra...")
    plot_sample_efficiency()

    print("\n4. Sensibilidade ao Learning Rate (α)...")
    plot_hyperparameter_sensitivity()

    print("\n" + "=" * 60)
    print(f"  Todas as figuras salvas em: {FIGURES_DIR}")
    print("=" * 60)
