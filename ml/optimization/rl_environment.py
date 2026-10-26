# ml/optimization/rl_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional

class AdAuctionEnv(gym.Env):
    """
    A simplified, fast environment for training RL agents.
    It mimics the dynamics of the full simulation but runs much faster
    to allow the agent to play millions of rounds.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, config=None):
        super(AdAuctionEnv, self).__init__()

        # --- Define Action Space ---
        # Action: A multiplier for the bid. 
        # 0 = Shade aggressively (50%), 1 = Base Bid, 2 = Aggressive (150%)
        # We map this to a continuous value or discrete buckets.
        # Using Discrete(5) for stability: 
        # 0: 0.5x, 1: 0.75x, 2: 1.0x, 3: 1.25x, 4: 1.5x
        self.action_space = spaces.Discrete(5)

        # --- Define Observation Space ---
        # What the agent sees:
        # [Budget Remaining %, Time Remaining %, Predicted CTR, Competition Level]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        # --- Internal State ---
        self.total_auctions = 1000
        self.current_step = 0
        self.initial_budget = 1000.0
        self.current_budget = self.initial_budget
        
        # Market simulation parameters
        self.base_ctr = 0.05
        self.value_per_click = 5.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_budget = self.initial_budget
        return self._get_obs(), {}

    def _get_obs(self):
        # Generate a random context for this auction
        self._user_value = np.random.uniform(0.01, 0.2) # Simulating Pred CTR
        self._competition = np.random.uniform(0.5, 2.0) # Simulating Market Heat
        
        budget_pct = self.current_budget / self.initial_budget
        time_pct = self.current_step / self.total_auctions
        
        return np.array([
            budget_pct, 
            time_pct, 
            self._user_value, 
            self._competition
        ], dtype=np.float32)

    def step(self, action):
        # Map action to multiplier
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
        bid_mult = multipliers[action]
        
        base_bid = 1.0
        my_bid = base_bid * bid_mult
        
        # --- Simulate Auction Outcome ---
        # Simplified logic: Higher competition + lower bid = lower win prob
        # Win prob formula (sigmoid-like)
        win_score = (my_bid * 0.5) - (self._competition * 0.3) + np.random.normal(0, 0.1)
        win_prob = 1 / (1 + np.exp(-win_score))
        
        won = np.random.random() < win_prob
        click = np.random.random() < self._user_value
        
        # --- Calculate Reward (Profit) ---
        reward = 0.0
        
        if won:
            # Second Price logic: pay approx market price (randomized)
            cost = base_bid * self._competition * 0.8 
            
            if cost <= self.current_budget:
                self.current_budget -= cost
                if click:
                    reward = self.value_per_click - cost
                else:
                    reward = -cost # Lost money on impression
            else:
                # Won but couldn't pay (should be rare due to pacing)
                reward = -10 # Penalty
                self.current_budget = 0
        else:
            # Didn't win, didn't pay
            reward = 0.0

        # Budget Penalty: Encourage saving money if ROI is bad? 
        # No, let reward handle it.

        self.current_step += 1
        terminated = self.current_budget < 0.1
        truncated = self.current_step >= self.total_auctions
        
        # Optional: Heavy penalty for finishing early with budget remaining
        if truncated and self.current_budget > (self.initial_budget * 0.5):
            reward -= 5 

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Budget: ${self.current_budget:.2f}")