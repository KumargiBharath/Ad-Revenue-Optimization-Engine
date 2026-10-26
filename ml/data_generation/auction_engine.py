from typing import List
import numpy as np
from .entities import Auction, Ad, User
from .click_model import ClickModel
from .config import GlobalConfig

class AuctionEngine:
    def __init__(self, config: GlobalConfig, click_model: ClickModel):
        self.config = config
        self.click_model = click_model

    def run_auction(
        self,
        auction: Auction,
        user: User,
        competing_ads: List[Ad],
        rng: np.random.Generator,
        position: str = "top",
        time_of_day: float = 0.5
    ) -> Auction:
        
        if not competing_ads:
            return auction

        # Limit competitors
        num_competitors = min(len(competing_ads), self.config.data_generation.competitors_per_auction)
        competitors = rng.choice(competing_ads, num_competitors, replace=False)

        # Rank by score (Bid * Quality)
        scores = []
        for ad in competitors:
            score = ad.bid_amount * ad.quality_score
            scores.append((ad, score))

        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)
        winner = scores[0][0]

        # FIX: Second-Price Auction Logic
        # Price paid = Bid of second place + small increment
        if len(scores) > 1:
            second_highest_bid = scores[1][0].bid_amount
            price_to_pay = second_highest_bid + 0.01
        else:
            # Fallback if only one competitor (pay reserve price or own bid)
            price_to_pay = winner.bid_amount 

        # Determine Click
        click_occurred = self.sample_click(winner, user, position, time_of_day, rng)

        # Update Auction Entity
        auction.competing_ads = list(competitors)
        auction.floor_price = price_to_pay # Store actual cost here
        
        # Resolve auction with actual cost
        auction.resolve_auction(winning_ad=winner, click_occurred=click_occurred, cost=price_to_pay)

        return auction

    def sample_click(self, ad: Ad, user: User, position: str, time_of_day: float, rng: np.random.Generator) -> bool:
        click_prob = self.click_model.compute_click_probability(
            user=user,
            ad=ad,
            position=position,
            time_of_day=time_of_day,
            rng=rng
        )
        return rng.random() < click_prob 