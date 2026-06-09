"""
AgentQLearning - Q-Learning tabular para gestao de portfolio.
"""

import numpy as np

from agentes.base_agent import BaseAgent


class AgentQLearning(BaseAgent):
    """Agente Q-Learning off-policy."""

    agent_name = "Q-Learning"

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> float:
        """Atualiza Q(s, a) usando target off-policy max_a Q(s', a)."""
        current_q = self.q_table[state][action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = td_target - current_q
        self.q_table[state][action] += self.alpha * td_error
        return float(td_error)

    def train(
        self,
        env,
        n_episodes: int = 500,
        log_interval: int = 50,
    ) -> dict:
        """Treina o agente por N episodios."""
        history = {
            "episode_rewards": [],
            "episode_portfolio_values": [],
            "episode_epsilons": [],
            "episode_alphas": [],
            "episode_td_errors": [],
        }

        print(f"\n{'='*60}")
        print(f"  Q-LEARNING TREINAMENTO - {n_episodes} episodios")
        print(
            f"  alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon}, bins={self.n_bins}"
        )
        print(f"{'='*60}\n")

        for episode in range(1, n_episodes + 1):
            state = self.discretize(env.reset())

            total_reward = 0.0
            total_td_error = 0.0
            n_steps = 0
            done = False
            info = {"portfolio_value": env.portfolio_value}

            while not done:
                action = self.choose_action(state)
                next_state_continuous, reward, done, info = env.step(action)
                next_state = self.discretize(next_state_continuous)

                td_error = self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                total_td_error += abs(td_error)
                n_steps += 1

            self.decay_epsilon()

            history["episode_rewards"].append(total_reward)
            history["episode_portfolio_values"].append(info["portfolio_value"])
            history["episode_epsilons"].append(self.epsilon)
            history["episode_alphas"].append(self.alpha)
            history["episode_td_errors"].append(
                total_td_error / max(n_steps, 1)
            )

            if episode % log_interval == 0 or episode == 1:
                avg_reward = np.mean(history["episode_rewards"][-log_interval:])
                print(
                    f"  Ep {episode:4d}/{n_episodes} | "
                    f"Reward: {total_reward:8.4f} | "
                    f"Avg({log_interval}): {avg_reward:8.4f} | "
                    f"Value: R$ {info['portfolio_value']:,.2f} | "
                    f"epsilon: {self.epsilon:.4f} | "
                    f"alpha: {self.alpha:.4f} | "
                    f"Q-states: {len(self.q_table):,}"
                )

        history["n_states_visited"] = len(self.q_table)

        print(f"\n{'='*60}")
        print("  TREINAMENTO CONCLUIDO")
        print(f"  Estados visitados: {len(self.q_table):,}")
        print(f"  epsilon final: {self.epsilon:.4f}")
        print(f"  alpha fixo: {self.alpha:.4f}")
        print(
            f"  Reward medio (ultimos 50): "
            f"{np.mean(history['episode_rewards'][-50:]):.4f}"
        )
        print(f"{'='*60}\n")

        return history
