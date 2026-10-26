from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
# 1
#To Run:-"python -m ml.data_generation.simulator"
from .config import load_config
from .entities import User, Advertiser, Ad, Auction
from .click_model import ClickModel
from .auction_engine import AuctionEngine
from .budget_manager import BudgetManager
from ..validation.schema import DatasetSchema

class AdAuctionSimulator:
    def __init__(self, project_root: str):
        self.config = load_config(project_root)
        print(f"📊 Simulating with Noise STD: {self.config.click_model.noise_std}")
        self.rng = np.random.default_rng(self.config.random_seed)
        self.interest_pool = ["sports", "tech", "fashion", "finance", "health", "gaming", "travel"]
        self.ad_categories = ["sports", "tech", "fashion", "finance", "health", "gaming", "travel"]
        self.click_model = ClickModel(self.config)
        self.auction_engine = AuctionEngine(self.config, self.click_model)
        self.advertisers = self._generate_advertisers()
        self.ads = self._generate_ads()
        self.budget_manager = BudgetManager(self.advertisers)
        num_users = max(100, int(self.config.data_generation.num_impressions * 0.1))
        self.users = self._generate_user_pool(num_users)

    def _generate_user_pool(self, num_users: int) -> List[User]:
        users = []
        for user_id in range(num_users):
            users.append(
                User.generate_random(
                    user_id=user_id,
                    interest_pool=self.interest_pool,
                    config_attributes=self.config.data_generation.user_attributes.__dict__,
                    rng=self.rng
                )
            )
        return users

    def _generate_advertisers(self):
        advertisers = []
        industries = ["tech", "retail", "finance", "health", "gaming", "travel"]
        for adv_id in range(self.config.data_generation.num_advertisers):
            advertisers.append(
                Advertiser.generate_random(
                    advertiser_id=adv_id,
                    industries=industries,
                    budget_range=(
                        self.config.data_generation.min_budget,
                        self.config.data_generation.max_budget
                    ),
                    rng=self.rng
                )
            )
        return advertisers

    def _generate_ads(self):
        ads = []
        ad_id = 0
        bid_range = (self.config.data_generation.min_bid, self.config.data_generation.max_bid)
        for adv in self.advertisers:
            num_ads = self.rng.integers(1, 4)
            for _ in range(num_ads):
                ads.append(
                    Ad.generate_random(
                        ad_id=ad_id,
                        advertiser_id=adv.advertiser_id,
                        categories=self.ad_categories, 
                        bid_range=bid_range,
                        rng=self.rng
                    )
                )
                ad_id += 1
        return ads

    def run(self):
        # Pre-allocate arrays (Massive Speedup)
        n = self.config.data_generation.num_impressions
        data = {
            "auction_id": np.zeros(n, dtype=int),
            "user_id": np.zeros(n, dtype=int),
            "ad_id": np.zeros(n, dtype=object),
            "advertiser_id": np.zeros(n, dtype=object),
            "age_group": np.zeros(n, dtype=object),
            "device_type": np.zeros(n, dtype=object),
            "location": np.zeros(n, dtype=object),
            "user_historical_ctr": np.zeros(n, dtype=float),
            "user_interests": np.zeros(n, dtype=object),
            "ad_category": np.zeros(n, dtype=object),
            "creative_type": np.zeros(n, dtype=object),
            "position": np.zeros(n, dtype=object),
            "bid": np.zeros(n, dtype=float),
            "quality_score": np.zeros(n, dtype=float),
            "ad_historical_ctr": np.zeros(n, dtype=float),
            "hour_of_day": np.zeros(n, dtype=int),
            "day_of_week": np.zeros(n, dtype=int),
            "publisher_category": np.zeros(n, dtype=object),
            "click": np.zeros(n, dtype=int),
            "revenue": np.zeros(n, dtype=float)
        }

        active_advertiser_ids = self.budget_manager.get_active_advertisers()
        num_users = len(self.users)

        for imp_id in tqdm(range(n), desc="Simulating Auctions"):
            # 1. Fast User Selection
            user_idx = self.rng.integers(0, num_users)
            user = self.users[user_idx]
            
            # 2. Fast Ad Selection
            active_ads = [ad for ad in self.ads if ad.advertiser_id in active_advertiser_ids]
            
            if not active_ads: break
            
            # Simplified Auction Logic for Speed
            winner = None
            click_occurred = False
            revenue = 0.0
            
            if active_ads:
                competitors = self.rng.choice(active_ads, min(len(active_ads), 3), replace=False)
                # Simple max score logic
                winner = max(competitors, key=lambda ad: ad.bid_amount * ad.quality_score)
                
                # Click Logic
                position = self.rng.choice(["top", "side", "bottom"], p=[0.5, 0.3, 0.2])
                hour = self.rng.integers(0, 24)
                prob = self.click_model.compute_click_probability(user, winner, position, hour/24.0, self.rng)
                click_occurred = self.rng.random() < prob
                
                if click_occurred:
                    revenue = winner.bid_amount # Simplified revenue
                    # Deduct budget logic (optional for speed, but good to keep)
                    if not self.budget_manager.deduct_budget(winner, winner.bid_amount):
                        active_advertiser_ids = self.budget_manager.get_active_advertisers()

            # Fill Arrays
            data['auction_id'][imp_id] = imp_id
            data['user_id'][imp_id] = user.user_id
            data['age_group'][imp_id] = user.age_group
            data['device_type'][imp_id] = user.device_type
            data['location'][imp_id] = user.location
            data['user_historical_ctr'][imp_id] = user.historical_ctr
            data['user_interests'][imp_id] = "|".join(user.interests)
            
            if winner:
                data['ad_id'][imp_id] = winner.ad_id
                data['advertiser_id'][imp_id] = winner.advertiser_id
                data['ad_category'][imp_id] = winner.category
                data['creative_type'][imp_id] = winner.creative_type
                data['bid'][imp_id] = winner.bid_amount
                data['quality_score'][imp_id] = winner.quality_score
                data['ad_historical_ctr'][imp_id] = winner.historical_ctr
            
            data['click'][imp_id] = int(click_occurred)
            data['revenue'][imp_id] = revenue
            data['hour_of_day'][imp_id] = self.rng.integers(0, 24)
            data['day_of_week'][imp_id] = self.rng.integers(0, 7)
            data['publisher_category'][imp_id] = self.rng.choice(self.ad_categories)
            data['position'][imp_id] = position

        df = pd.DataFrame(data)
        df = DatasetSchema.enforce_schema(df)
        DatasetSchema.validate_dataframe(df)
        return df

    def save(self, df: pd.DataFrame, path: str = "simulated_ad_auction_dataset.csv"):
        df.to_csv(path, index=False)
        print(f"Dataset saved to {path} | Shape: {df.shape}")
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Get project root (assuming script is run from project root or via module)
    project_root = Path(__file__).resolve().parents[2]
    
    # Define the specific directory and path
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    output_path = data_dir / "simulated_ad_auction_dataset.csv"
    
    print("Initializing Simulator...")
    simulator = AdAuctionSimulator(str(project_root))
    
    print("Running Simulation...")
    df = simulator.run()
    
    print("Saving Dataset...")
    # Pass the full path to the save function
    simulator.save(df, path=str(output_path))
    
    print("Done.")