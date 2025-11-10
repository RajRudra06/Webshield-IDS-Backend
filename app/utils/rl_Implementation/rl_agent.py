import numpy as np
import pickle
import random
from collections import defaultdict
from pathlib import Path

class URLPredictionRLAgent:
    
    def __init__(self, n_actions=5, learning_rate=0.1, discount_factor=0.95, epsilon=0.15):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions
        
        # Action mapping
        self.action_names = {
            0: "trust_consensus_override",
            1: "trust_meta_model",
            2: "trust_highest_confidence_base",
            3: "weighted_average",
            4: "unanimous_high_conf_only"
        }
        
        # Load existing Q-table if available
        self.model_path = Path(__file__).parent.parent.parent / "models" / "rl_agent.pkl"
        self.load()
    
    def select_action(self, state_key, explore=True):
    
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        
        return int(np.argmax(self.q_table[state_key]))
    
    def update(self, state_key, action, reward, next_state_key):
     
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        
        # Q(s,a) = Q(s,a) + lr * (reward + gamma * max(Q(s',a')) - Q(s,a))
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def save(self):
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(dict(self.q_table), f)
            print(f"✅ RL agent saved to {self.model_path}")
        except Exception as e:
            print(f"⚠️  Failed to save RL agent: {e}")
    
    def load(self):
        try:
            if self.model_path.exists():
                with open(self.model_path, 'rb') as f:
                    loaded_table = pickle.load(f)
                    self.q_table = defaultdict(
                        lambda: np.zeros(self.n_actions),
                        loaded_table
                    )
                print(f"✅ RL agent loaded from {self.model_path}")
            else:
                print("ℹ️  No existing RL agent found, starting fresh")
        except Exception as e:
            print(f"⚠️  Failed to load RL agent: {e}")
    
    def get_stats(self):
        return {
            "total_states_explored": len(self.q_table),
            "epsilon": self.epsilon,
            "learning_rate": self.lr,
            "discount_factor": self.gamma,
            "actions": self.action_names
        }
    
    def get_q_values_sample(self, n=5):
        sample = {}
        for i, (state, q_vals) in enumerate(list(self.q_table.items())[:n]):
            sample[f"state_{i}"] = {
                "state": str(state),
                "q_values": {
                    self.action_names[j]: float(q) 
                    for j, q in enumerate(q_vals)
                },
                "best_action": self.action_names[int(np.argmax(q_vals))]
            }
        return sample


# Global RL agent instance
rl_agent = URLPredictionRLAgent()