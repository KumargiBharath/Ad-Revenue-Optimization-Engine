from typing import Dict, List
from collections import defaultdict
import numpy as np

class FrequencyCapper:
    """
    Tracks how many times a specific user has seen a specific ad.
    Implements Frequency Capping to prevent Ad Fatigue.
    """
    def __init__(self, max_views: int = 3):
        self.max_views = max_views
        self.view_history = defaultdict(lambda: defaultdict(int))

    def is_eligible(self, user_id: int, ad_id: int) -> bool:
        return self.view_history[user_id][ad_id] < self.max_views

    def record_impression(self, user_id: int, ad_id: int):
        self.view_history[user_id][ad_id] += 1

class UserValueProfiler:
    """
    Calculates a multiplier based on user quality.
    """
    def __init__(self, high_value_threshold: float = 0.15, boost_factor: float = 1.2):
        self.threshold = high_value_threshold
        self.boost = boost_factor

    def get_value_multiplier(self, user_historical_ctr: float) -> float:
        if user_historical_ctr >= self.threshold:
            return self.boost
        return 1.0

class BudgetPacer:
    """
    Simple Probabilistic Pacer.
    """
    def __init__(self, total_auctions: int, target_spend_ratio: float = 0.95):
        self.total_auctions = total_auctions
        self.target_spend_ratio = target_spend_ratio 

    def should_participate(self, current_auction_idx: int, budget_remaining: float, initial_budget: float) -> bool:
        if initial_budget == 0: return False
        progress = current_auction_idx / self.total_auctions
        target_spend_so_far = initial_budget * progress * self.target_spend_ratio
        actual_spend_so_far = initial_budget - budget_remaining
        
        if actual_spend_so_far < target_spend_so_far:
            return True 
        return False

# --- NEW FEATURES ---

class MarketHeatAnalyzer:
    """
    Adjusts Reserve Price based on Demand/Supply.
    If many advertisers compete for few slots -> Price Floor GOES UP.
    """
    def __init__(self, base_reserve: float = 0.50, intensity: float = 0.5):
        self.base_reserve = base_reserve
        self.intensity = intensity

    def calculate_dynamic_reserve(self, num_candidates: int, num_slots: int) -> float:
        if num_candidates <= num_slots:
            return self.base_reserve
        
        # Competition Ratio (e.g., 10 candidates for 3 slots = 3.33)
        ratio = num_candidates / num_slots
        
        # Increase floor based on competition
        # Formula: Base + (Ratio * Intensity)
        dynamic_floor = self.base_reserve + (ratio * self.intensity)
        
        # Cap at reasonable limit (e.g., $5.00)
        return min(dynamic_floor, 5.0)

class DiversityManager:
    """
    Penalizes ads that have won too frequently recently.
    Promotes market diversity and prevents monopolies.
    """
    def __init__(self, penalty_factor: float = 0.05, window_size: int = 50):
        self.penalty_factor = penalty_factor
        self.window_size = window_size
        self.recent_winners = [] # Sliding window of IDs

    def get_diversity_multiplier(self, ad_id: int) -> float:
        """Returns a multiplier < 1.0 if this ad won recently."""
        if not self.recent_winners:
            return 1.0
        
        count = self.recent_winners.count(ad_id)
        # Linear penalty: subtract penalty_factor * count
        multiplier = 1.0 - (count * self.penalty_factor)
        return max(multiplier, 0.5) # Don't penalize more than 50%

    def record_win(self, ad_id: int):
        self.recent_winners.append(ad_id)
        if len(self.recent_winners) > self.window_size:
            self.recent_winners.pop(0)