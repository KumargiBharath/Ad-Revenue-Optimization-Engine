# ml/optimization/strategic_agents.py
import numpy as np
from typing import Dict, Optional
from collections import deque

class BidStrategyAgent:
    """
    Base class for strategic advertisers.
    """
    def __init__(self, base_bid: float, agent_id: int):
        self.base_bid = base_bid
        self.agent_id = agent_id
    
    def get_bid(self, context: Dict) -> float:
        """
        Returns the bid amount for this specific auction.
        Context can include: user_value, competition_estimate, time_remaining, etc.
        """
        return self.base_bid

    def update_state(self, won: bool, price_paid: float, clicks: int, cost: float):
        """
        Feedback loop: Called after auction results to update internal strategy.
        """
        pass

class StaticAgent(BidStrategyAgent):
    """
    Preserves the legacy behavior (Bid = Base Bid).
    """
    pass

class BidShadingAgent(BidStrategyAgent):
    """
    Strategic Agent: Tries to pay the minimum necessary to win.
    Strategy: If winning often, lower the bid to save money.
    Exploits Second-Price auctions.
    """
    def __init__(self, base_bid: float, agent_id: int, aggressiveness: float = 0.05):
        super().__init__(base_bid, agent_id)
        self.aggressiveness = aggressiveness
        self.win_history = deque(maxlen=20) # Sliding window of wins (1) and losses (0)
        self.current_modifier = 1.0

    def get_bid(self, context: Dict) -> float:
        # Apply the learned modifier
        return self.base_bid * self.current_modifier

    def update_state(self, won: bool, price_paid: float, clicks: int, cost: float):
        self.win_history.append(1 if won else 0)
        
        if len(self.win_history) < 5: return # Wait for data

        win_rate = sum(self.win_history) / len(self.win_history)
        
        # Logic:
        # If win_rate is very high (> 70%), we are overbidding or dominating. Lower bid.
        # If win_rate is low (< 30%), we are losing. Raise bid.
        
        if win_rate > 0.7:
            self.current_modifier = max(0.5, self.current_modifier - self.aggressiveness)
        elif win_rate < 0.3:
            self.current_modifier = min(1.5, self.current_modifier + self.aggressiveness)

class ROIOptimizingAgent(BidStrategyAgent):
    """
    Strategic Agent: Adjusts bids based on Return on Investment (ROI).
    If ROI is high (profitable), bid more aggressively. If low, pull back.
    """
    def __init__(self, base_bid: float, agent_id: int, target_roi: float = 1.5):
        super().__init__(base_bid, agent_id)
        self.target_roi = target_roi
        self.total_revenue = 0.0
        self.total_cost = 0.0
        self.current_modifier = 1.0

    def get_bid(self, context: Dict) -> float:
        return self.base_bid * self.current_modifier

    def update_state(self, won: bool, price_paid: float, clicks: int, cost: float):
        # This agent needs actual ROI data. 
        # In a real system, revenue = clicks * value_per_click.
        # For simulation, we assume value_per_click is approx known (e.g., $5.00)
        
        if won:
            # Simplified ROI calculation
            estimated_value = clicks * 5.0 # Placeholder value
            self.total_revenue += estimated_value
            self.total_cost += cost
            
            if self.total_cost > 0:
                current_roi = self.total_revenue / self.total_cost
                
                # Adjust Modifier
                if current_roi > self.target_roi:
                    # Making money? Bid more to get more volume
                    self.current_modifier = min(1.5, self.current_modifier + 0.05)
                else:
                    # Losing money? Reduce bid
                    self.current_modifier = max(0.5, self.current_modifier - 0.05)

# Factory Pattern for easy creation
def create_agent(strategy: str, base_bid: float, agent_id: int, **kwargs) -> BidStrategyAgent:
    if strategy == "static":
        return StaticAgent(base_bid, agent_id)
    elif strategy == "bid_shading":
        return BidShadingAgent(base_bid, agent_id, **kwargs)
    elif strategy == "roi_optimizer":
        return ROIOptimizingAgent(base_bid, agent_id, **kwargs)
    else:
        return StaticAgent(base_bid, agent_id)