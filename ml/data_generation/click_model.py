from typing import Optional
import numpy as np
from .entities import User, Ad
from .config import GlobalConfig

class ClickModel:
    """
    Simulates clicks for ads based on features and config weights.
    UPGRADED: Implements Non-Linear Ground Truth for higher model ceiling.
    """
    def __init__(self, config: GlobalConfig):
        self.config = config
        self.position_weights = config.data_generation.position_bias
        self.weights = config.click_model

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def compute_click_probability(
        self,
        user: User,
        ad: Ad,
        position: str = "top",
        time_of_day: Optional[float] = None,
        rng: np.random.Generator = None
    ) -> float:
        
        position_weight = getattr(self.position_weights, position, 1.0)
        
        # --- UPGRADE: ADVANCED GROUND TRUTH LOGIC ---
        
        # 1. Base Linear Score
        linear_score = (
            -4.5 # Base bias to maintain ~8% CTR
            + self.weights.quality_weight * ad.quality_score
            + self.weights.historical_ctr_weight * ad.historical_ctr
            + self.weights.position_weight * position_weight
            + self.weights.match_weight * self._interest_match(user, ad)
        )

        # 2. NON-LINEAR INTERACTION (The "Hidden" Pattern)
        # Users click significantly more if BOTH quality is high AND there is an interest match.
        interaction_boost = 0.0
        if self._interest_match(user, ad) > 0 and ad.quality_score > 0.7:
            interaction_boost = 1.5  # Strong boost

        # 3. SATURATION (Diminishing Returns)
        # Bids over a certain amount don't help as much
        bid_effect = np.log1p(ad.bid_amount) * self.weights.quality_weight * 0.5
        
        # 4. TEMPORAL DYNAMICS
        # Users on Mobile click more in evenings (0.7-0.9)
        time_effect = 0.0
        if user.device_type == "mobile" and 0.6 < time_of_day < 0.9:
            time_effect = 0.4
        
        # Combine
        score = linear_score + interaction_boost + bid_effect + time_effect
        
        # Add Noise (Reduced slightly to allow model to learn patterns)
        current_rng = rng if rng is not None else np.random.default_rng()
        noise = current_rng.normal(0, self.weights.noise_std * 0.8) 
        score += noise

        prob = self._sigmoid(score)
        return float(np.clip(prob, 0.0, 1.0))

    def _interest_match(self, user: User, ad: Ad) -> float:
        ad_category = ad.category.lower()
        matches = sum(1 for interest in user.interests if interest.lower() == ad_category)
        return float(matches > 0)