import numpy as np
import pickle
import random
from collections import defaultdict
from pathlib import Path

_rl_agent = None  # module-level singleton


class URLPredictionRLAgent:
    def __init__(self, n_actions=5, learning_rate=0.1, discount_factor=0.95, epsilon=0.15):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions

        self.action_names = {
            0: "trust_consensus_override",
            1: "trust_meta_model",
            2: "trust_highest_confidence_base",
            3: "weighted_average",
            4: "unanimous_high_conf_only",
        }

        self.model_path = (
            Path(__file__).parent.parent.parent / "models" / "rl_agent.pkl"
        )

        self._loaded = False

    def _load_if_needed(self):
        if self._loaded:
            return

        try:
            if self.model_path.exists():
                with open(self.model_path, "rb") as f:
                    loaded_table = pickle.load(f)
                    self.q_table = defaultdict(
                        lambda: np.zeros(self.n_actions),
                        loaded_table,
                    )
                print(f"✅ RL agent loaded from {self.model_path}")
            else:
                print("ℹ️  No existing RL agent found, starting fresh")
        except Exception as e:
            print(f"⚠️  Failed to load RL agent: {e}")

        self._loaded = True

    def select_action(self, state_key, explore=True):
        self._load_if_needed()

        if explore and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key):
        self._load_if_needed()

        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.lr * (
            reward + self.gamma * max_next_q - current_q
        )
        self.q_table[state_key][action] = new_q

    def save(self):
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(dict(self.q_table), f)
            print(f"✅ RL agent saved to {self.model_path}")
        except Exception as e:
            print(f"⚠️  Failed to save RL agent: {e}")

    def get_stats(self):
        return {
            "total_states_explored": len(self.q_table),
            "epsilon": self.epsilon,
            "learning_rate": self.lr,
            "discount_factor": self.gamma,
            "actions": self.action_names,
        }


def get_rl_agent():
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = URLPredictionRLAgent()
    return _rl_agent
