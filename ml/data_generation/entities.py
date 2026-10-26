from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

# ==============================
# USER ENTITY
# ==============================
@dataclass
class User:
    user_id: int
    age_group: str
    device_type: str
    location: str
    interests: List[str]
    historical_ctr: float

    @staticmethod
    def generate_random(
        user_id: int, 
        interest_pool: List[str], 
        config_attributes: Dict[str, List[str]], 
        rng: np.random.Generator
    ):
        return User(
            user_id=user_id,
            age_group=rng.choice(config_attributes['age_groups']),
            device_type=rng.choice(config_attributes['devices']),
            location=rng.choice(config_attributes['locations']),
            interests=list(rng.choice(interest_pool, size=rng.integers(1, 4), replace=False)),
            historical_ctr=rng.uniform(0.01, 0.2),
        )

# ==============================
# ADVERTISER ENTITY
# ==============================
@dataclass
class Advertiser:
    advertiser_id: int
    industry: str
    campaign_budget: float
    bid_strategy: str
    quality_score: float

    @staticmethod
    def generate_random(
        advertiser_id: int, 
        industries: List[str], 
        budget_range: tuple, 
        rng: np.random.Generator
    ):
        return Advertiser(
            advertiser_id=advertiser_id,
            industry=rng.choice(industries),
            campaign_budget=rng.uniform(budget_range[0], budget_range[1]),
            bid_strategy=rng.choice(["manual", "automatic", "target_cpa"]),
            quality_score=rng.uniform(0.5, 1.0),
        )

# ==============================
# AD ENTITY
# ==============================
@dataclass
class Ad:
    ad_id: int
    advertiser_id: int
    category: str
    creative_type: str
    historical_ctr: float
    quality_score: float
    bid_amount: float 

    @staticmethod
    def generate_random(
        ad_id: int, 
        advertiser_id: int, 
        categories: List[str], 
        bid_range: tuple,
        rng: np.random.Generator
    ):
        return Ad(
            ad_id=ad_id,
            advertiser_id=advertiser_id,
            category=rng.choice(categories),
            creative_type=rng.choice(["banner", "video", "text"]),
            historical_ctr=rng.uniform(0.01, 0.2),
            quality_score=rng.uniform(0.5, 1.0),
            bid_amount=rng.uniform(bid_range[0], bid_range[1])
        )

# ==============================
# AUCTION ENTITY
# ==============================
@dataclass
class Auction:
    auction_id: int
    timestamp: Optional[str] = None
    competing_ads: List[Ad] = field(default_factory=list)
    winning_ad: Optional[Ad] = None
    floor_price: float = 0.0
    click_occurred: Optional[bool] = None
    revenue: Optional[float] = None

    def resolve_auction(self, winning_ad: Ad, click_occurred: bool, cost: float = 0.0):
        self.winning_ad = winning_ad
        self.click_occurred = click_occurred
        
        if click_occurred:
            self.revenue = cost # Use the calculated second-price
        else:
            self.revenue = 0.0