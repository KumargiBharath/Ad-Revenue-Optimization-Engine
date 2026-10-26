from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
import pandas as pd
import numpy as np

from ml.data_generation.entities import Ad

@dataclass
class SlotResult:
    slot_id: int
    winner_ad: Optional[Ad]
    winning_bid: float
    price_paid: float
    predicted_revenue: float
    rank_score: float

@dataclass
class AuctionResult:
    slots: List[SlotResult] = field(default_factory=list)
    strategy_used: str = ""
    status: str = "success"
    
    @property
    def winner_ad(self):
        return self.slots[0].winner_ad if self.slots else None

class AuctionOptimizer:
    def __init__(
        self, 
        ranking_strategy: Literal["revenue", "ctr", "bid_only"] = "revenue",
        pricing_mechanism: Literal["first_price", "second_price"] = "second_price",
        reserve_price: float = 0.10,
        max_price_cap: float = 100.0,
        min_ctr_threshold: float = 0.001,
        num_slots: int = 3,
        soft_floor_ratio: float = 0.0,
        use_dynamic_reserve: bool = False, # NEW
        use_diversity: bool = False # NEW
    ):
        self.ranking_strategy = ranking_strategy
        self.pricing_mechanism = pricing_mechanism
        self.reserve_price = reserve_price
        self.max_price_cap = max_price_cap
        self.min_ctr_threshold = min_ctr_threshold
        self.num_slots = num_slots
        self.soft_floor_ratio = soft_floor_ratio
        self.use_dynamic_reserve = use_dynamic_reserve
        self.use_diversity = use_diversity

    # MODIFY THIS METHOD
    def _calculate_scores(
        self, 
        candidates: List[Dict], 
        ctr_predictions: np.ndarray,
        diversity_mults: List[float] = None,
        custom_bids: Dict[int, float] = None # NEW: Optional override bids
    ) -> pd.DataFrame:
        
        data = []
        for i, cand in enumerate(candidates):
            ad = cand['ad']
            pred_ctr = ctr_predictions[i]
            div_mult = diversity_mults[i] if diversity_mults else 1.0
            
            # LOGIC: Use custom bid if provided, else fallback to ad.bid_amount
            effective_bid = ad.bid_amount
            if custom_bids and ad.ad_id in custom_bids:
                effective_bid = custom_bids[ad.ad_id]

            if pred_ctr < self.min_ctr_threshold:
                rank_score = -1.0
            else:
                if self.ranking_strategy == "revenue":
                    score = pred_ctr * effective_bid # Use effective_bid
                    rank_score = score * div_mult 
                elif self.ranking_strategy == "bid_only":
                    score = effective_bid
                    rank_score = score * div_mult 
                elif self.ranking_strategy == "ctr":
                    score = pred_ctr
                    rank_score = score * div_mult
                else:
                    rank_score = 0.0

            data.append({
                "ad": ad,
                "pred_ctr": pred_ctr,
                "bid": effective_bid, # Store effective_bid
                "rank_score": rank_score
            })
            
        return pd.DataFrame(data)

    # MODIFY THIS METHOD SIGNATURE
    def run_auction(
        self, 
        candidates: List[Dict], 
        ctr_predictions: np.ndarray,
        user_value_multiplier: float = 1.0,
        dynamic_reserve_price: float = None,
        diversity_mults: List[float] = None,
        custom_bids: Dict[int, float] = None # NEW PARAMETER
    ) -> AuctionResult:
        
        if not candidates:
            return AuctionResult(status="no_candidates", strategy_used=self.ranking_strategy)

        # Pass custom_bids down
        df = self._calculate_scores(candidates, ctr_predictions, diversity_mults, custom_bids)
        
        # ... REST OF THE METHOD REMAINS EXACTLY THE SAME ...
        # (No changes needed to pricing logic, it reads from the df which now has the dynamic bid)
        
        df_valid = df[df['rank_score'] >= 0].sort_values(by="rank_score", ascending=False).reset_index(drop=True)
        
        if df_valid.empty:
            return AuctionResult(status="below_quality", strategy_used=self.ranking_strategy)

        slots_result = []
        num_available_slots = min(self.num_slots, len(df_valid))
        
        current_reserve = self.reserve_price
        if self.use_dynamic_reserve and dynamic_reserve_price is not None:
            current_reserve = dynamic_reserve_price

        for i in range(num_available_slots):
            current_row = df_valid.iloc[i]
            winner_ad = current_row['ad']
            winner_pred_ctr = current_row['pred_ctr']
            
            raw_price = 0.0
            
            if self.pricing_mechanism == "first_price":
                raw_price = current_row['bid'] # Uses effective bid
            
            elif self.pricing_mechanism == "second_price":
                raw_price = current_reserve
                
                if i + 1 < len(df_valid):
                    next_row = df_valid.iloc[i+1]
                    
                    if self.ranking_strategy == "revenue":
                        if winner_pred_ctr > 0:
                            # Pricing logic remains robust because it relies on the 'bid' column in df
                            raw_price = next_row['rank_score'] / winner_pred_ctr
                
                else:
                    raw_price = current_reserve

            user_floor = current_reserve * user_value_multiplier
            soft_floor = 0.0
            if self.soft_floor_ratio > 0:
                soft_floor = current_row['bid'] * self.soft_floor_ratio
            
            final_floor = max(user_floor, soft_floor, current_reserve)
            final_price = max(raw_price, final_floor)
            final_price = min(final_price, current_row['bid'], self.max_price_cap) # max_price_cap check uses effective bid
            final_price = np.round(final_price, 2)

            predicted_revenue = final_price * winner_pred_ctr

            slots_result.append(SlotResult(
                slot_id=i,
                winner_ad=winner_ad,
                winning_bid=current_row['bid'],
                price_paid=final_price,
                predicted_revenue=predicted_revenue,
                rank_score=current_row['rank_score']
            ))

        return AuctionResult(
            slots=slots_result,
            strategy_used=self.ranking_strategy,
            status="success"
        )