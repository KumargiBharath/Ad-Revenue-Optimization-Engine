import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from copy import deepcopy
from collections import defaultdict
# 3
# To Run this:- "python -m ml.pipelines.business_optimization_pipeline"
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ml.data_generation.config import load_config
from ml.data_generation.simulator import AdAuctionSimulator
from ml.data_generation.budget_manager import BudgetManager
from ml.models.ctr_system import CTRSystem
from ml.optimization.auction_optimizer import AuctionOptimizer
from ml.optimization.business_intelligence import (
    FrequencyCapper, UserValueProfiler, BudgetPacer, 
    MarketHeatAnalyzer, DiversityManager
)
from ml.data_generation.click_model import ClickModel

# --- PHASE 14 & 15 DISABLED ---
# The old AuctionOptimizer does not support 'custom_bids', so these strategies are disabled.
# from ml.optimization.strategic_agents import create_agent

# --- RL IMPORT (ROBUST) ---
RL_AVAILABLE = False
DQN = None
try:
    from stable_baselines3 import DQN
    RL_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Warning: Could not load Stable-Baselines3/PyTorch. Error: {e}")

class ThompsonSampler:
    def __init__(self, effective_sample_size: float = 100.0):
        self.n = effective_sample_size

    def sample(self, pred_ctrs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        alphas = (pred_ctrs * self.n) + 1
        betas = ((1 - pred_ctrs) * self.n) + 1
        return rng.beta(alphas, betas)

def run_revenue_simulation():
    print("--- 🚀 Phase 20: BUSINESS OPTIMIZATION (Legacy Compatibility Mode) ---")
    
    config = load_config(str(project_root))
    model_path = project_root / "artifacts" / "models" / "ctr_system_v1.pkl"
    
    if not model_path.exists():
        print("❌ Error: Trained model not found.")
        return

    ctr_system = CTRSystem.load(str(model_path))
    click_model = ClickModel(config)
    sim_env = AdAuctionSimulator(str(project_root))
    rng = np.random.default_rng(42)
    
    thompson_sampler = ThompsonSampler(effective_sample_size=50.0)

    # Components
    freq_capper = FrequencyCapper(max_views=3)
    value_profiler = UserValueProfiler(high_value_threshold=0.15, boost_factor=1.5)
    heat_analyzer = MarketHeatAnalyzer(base_reserve=0.50, intensity=0.5)
    diversity_mgr = DiversityManager(penalty_factor=0.05, window_size=50)
    
    num_sim_auctions = 20000 
    candidates_per_auction = 30 
    
    avg_bid = np.mean([ad.bid_amount for ad in sim_env.ads])
    value_per_click = avg_bid * 1.5 
    
    print(f"💡 Assumption: Click Value = ${value_per_click:.2f}")
    
    budget_mgr_baseline = BudgetManager(sim_env.advertisers)
    budget_mgr_ml = BudgetManager(sim_env.advertisers)
    
    initial_budgets = budget_mgr_baseline.get_state()
    
    pacer_baseline = BudgetPacer(total_auctions=num_sim_auctions)
    pacer_ml = BudgetPacer(total_auctions=num_sim_auctions)

    # --- STRATEGIES (Compatible with old optimizer) ---
    # Note: use_dynamic_reserve and use_diversity flags removed as they don't exist in old __init__
    
    strategies = {
        "Baseline (Bid Only)": AuctionOptimizer(
            ranking_strategy="bid_only", 
            reserve_price=0.50,
            num_slots=3
        ),
        "ML Optimized (Volume)": AuctionOptimizer(
            ranking_strategy="revenue", 
            pricing_mechanism="second_price",
            reserve_price=0.50,
            min_ctr_threshold=0.01,
            num_slots=3,
            soft_floor_ratio=0.70
        ),
        "ML Optimized (Margin)": AuctionOptimizer(
            ranking_strategy="revenue", 
            pricing_mechanism="second_price",
            reserve_price=0.50,
            min_ctr_threshold=0.05,
            num_slots=3,
            soft_floor_ratio=0.50
        ),
    }
    
    budget_managers = {
        "Baseline (Bid Only)": budget_mgr_baseline,
        "ML Optimized (Volume)": budget_mgr_ml,
        "ML Optimized (Margin)": deepcopy(budget_mgr_ml),
    }
    
    pacers = {
        "Baseline (Bid Only)": pacer_baseline,
        "ML Optimized (Volume)": pacer_ml,
        "ML Optimized (Margin)": deepcopy(pacer_ml),
    }
    
    results = {k: {
        "revenue": 0.0, "clicks": 0, "impressions": 0, "fatigue_skips": 0, 
        "advertiser_value": 0.0, 
        "mobile_revenue": 0.0, "desktop_revenue": 0.0,
        "high_value_revenue": 0.0, "low_value_revenue": 0.0
    } for k in strategies.keys()}
    
    print(f"🔬 Simulating {num_sim_auctions} auctions...")
    
    # --- STEP 1: PRE-GENERATE CONTEXT ---
    user_indices = rng.integers(0, len(sim_env.users), num_sim_auctions)
    hours = rng.integers(0, 24, num_sim_auctions)
    days = rng.integers(0, 7, num_sim_auctions)
    publishers = rng.choice(sim_env.ad_categories, num_sim_auctions)
    
    # --- STEP 2: PRE-CALCULATE PREDICTIONS ---
    print("⚡ Building Super-Batch for Prediction...")
    
    batch_data = []
    auction_map = []
    
    ad_pool_indices = rng.integers(0, len(sim_env.ads), (num_sim_auctions, candidates_per_auction))

    for i in range(num_sim_auctions):
        user = sim_env.users[user_indices[i]]
        current_ad_indices = ad_pool_indices[i]
        
        for ad_idx in current_ad_indices:
            ad = sim_env.ads[ad_idx]
            batch_data.append({
                "auction_id": i, 
                "user_id": user.user_id, 
                "ad_id": ad.ad_id, 
                "advertiser_id": ad.advertiser_id,
                "age_group": user.age_group, 
                "device_type": user.device_type, 
                "location": user.location,
                "user_historical_ctr": user.historical_ctr, 
                "user_interests": "|".join(user.interests),
                "ad_category": ad.category, 
                "creative_type": ad.creative_type, 
                "position": "top", 
                "bid": ad.bid_amount, 
                "quality_score": ad.quality_score, 
                "ad_historical_ctr": ad.historical_ctr,
                "hour_of_day": hours[i],
                "day_of_week": days[i],
                "publisher_category": publishers[i]
            })
            auction_map.append((i, ad_idx))

    df_super = pd.DataFrame(batch_data)
    start_pred = time.time()
    print(f"   Predicting {len(df_super)} pairs...")
    super_probs = ctr_system.predict(df_super)['click_probability'].values
    print(f"   ⚡ Prediction Time: {time.time() - start_pred:.2f}s")
    
    pred_lookup = {}
    for row_idx, (auc_idx, ad_idx) in enumerate(auction_map):
        pred_lookup[(auc_idx, ad_idx)] = super_probs[row_idx]

    # --- STEP 3: SIMULATION LOOP ---
    print("🔄 Running Simulation...")
    
    start_sim = time.time()
    
    for i in tqdm(range(num_sim_auctions), desc="Simulating"):
        
        user = sim_env.users[user_indices[i]]
        current_hour = hours[i]
        current_ad_indices = ad_pool_indices[i]
        
        for name, optimizer in strategies.items():
            budget_mgr = budget_managers[name]
            pacer = pacers[name]
            
            active_advertiser_ids = budget_mgr.get_active_advertisers()
            
            eligible_ads = []
            pred_ctrs = []
            diversity_mults = []
            # custom_bids removed for compatibility
            
            for ad_idx in current_ad_indices:
                ad = sim_env.ads[ad_idx]
                prob = pred_lookup.get((i, ad_idx), 0.0)
                
                if ad.advertiser_id not in active_advertiser_ids: continue
                if budget_mgr.advertiser_budget[ad.advertiser_id] < ad.bid_amount: continue
                
                if not pacer.should_participate(
                    current_auction_idx=i,
                    budget_remaining=budget_mgr.advertiser_budget[ad.advertiser_id],
                    initial_budget=initial_budgets['budget'][ad.advertiser_id]
                ): continue
                
                if not freq_capper.is_eligible(user.user_id, ad.ad_id):
                    results[name]["fatigue_skips"] += 1
                    continue
                
                eligible_ads.append(ad)
                pred_ctrs.append(prob)
                diversity_mults.append(diversity_mgr.get_diversity_multiplier(ad.ad_id))
            
            if not eligible_ads: continue
            
            pred_ctrs = np.array(pred_ctrs)
            
            # Apply heuristic shading for the old "Margin" strategy
            if name == "ML Optimized (Margin)":
                pred_ctrs = pred_ctrs * 0.85 

            dynamic_reserve = heat_analyzer.calculate_dynamic_reserve(len(eligible_ads), 3)
            user_val_multiplier = value_profiler.get_value_multiplier(user.historical_ctr)
            
            candidates_input = [{"ad": ad} for ad in eligible_ads]
            
            # custom_bids argument removed from call
            result = optimizer.run_auction(
                candidates_input, 
                pred_ctrs, 
                user_value_multiplier=user_val_multiplier,
                dynamic_reserve_price=dynamic_reserve, 
                diversity_mults=diversity_mults
            )
            
            if result.status != "success": continue
            
            position_map = {0: "top", 1: "side", 2: "bottom"}
            
            for slot_res in result.slots:
                position_name = position_map.get(slot_res.slot_id, "bottom")
                
                freq_capper.record_impression(user.user_id, slot_res.winner_ad.ad_id)
                diversity_mgr.record_win(slot_res.winner_ad.ad_id)
                
                actual_click_prob = click_model.compute_click_probability(
                    user=user, 
                    ad=slot_res.winner_ad, 
                    position=position_name, 
                    time_of_day=current_hour / 24.0, 
                    rng=rng
                )
                actual_click = rng.random() < actual_click_prob
                
                if actual_click:
                    cost = slot_res.price_paid
                    success = budget_mgr.deduct_budget(slot_res.winner_ad, cost)
                    if success:
                        results[name]["revenue"] += cost
                        results[name]["clicks"] += 1
                        results[name]["advertiser_value"] += value_per_click
                        
                        if user.device_type == 'mobile':
                            results[name]["mobile_revenue"] += cost
                        else:
                            results[name]["desktop_revenue"] += cost
                            
                        if user.historical_ctr > 0.15:
                            results[name]["high_value_revenue"] += cost
                        else:
                            results[name]["low_value_revenue"] += cost
                
                results[name]["impressions"] += 1

    end_sim = time.time()
    duration = end_sim - start_sim
                
    # Reporting
    print("\n" + "="*80)
    print(f"{'Strategy':<30} | {'Plat. Revenue':<12} | {'Clicks':<8} | {'Impressions':<12} | {'CTR':<10}")
    print("-" * 80)
    
    baseline_rev = 0.0
    for name, data in results.items():
        if "Baseline" in name: baseline_rev = data['revenue']
        ctr = (data['clicks'] / data['impressions'] * 100) if data['impressions'] > 0 else 0
        print(f"{name:<30} | ${data['revenue']:<11.2f} | {data['clicks']:<8} | {data['impressions']:<12} | {ctr:.2f}%")
    
    print("="*80)
    
    # --- ROI ANALYSIS ---
    print(f"\n📊 UNIT ECONOMICS ANALYSIS")
    print(f"   Assumed Click Value: ${value_per_click:.2f}")
    print("-" * 80)
    print(f"{'Strategy':<30} | {'Spend':<12} | {'Value Gen.':<12} | {'Profit':<12} | {'ROI %':<15}")
    print("-" * 80)
    
    for name, data in results.items():
        spend = data['revenue'] 
        value_gen = data['advertiser_value']
        profit = value_gen - spend
        roi = (profit / spend * 100) if spend > 0 else 0
        print(f"{name:<30} | ${spend:<11.2f} | ${value_gen:<11.2f} | ${profit:<11.2f} | {roi:<14.2f}%")
        
    # --- SEGMENT BREAKDOWN ---
    print(f"\n🎯 REVENUE ATTRIBUTION (ML Margin)")
    print("-" * 80)
    ml_data = results["ML Optimized (Margin)"]
    if ml_data['revenue'] > 0:
        total_rev = ml_data['revenue']
        print(f"📱 Device Breakdown:")
        print(f"   Mobile:  ${ml_data['mobile_revenue']:.2f} ({ml_data['mobile_revenue']/total_rev*100:.1f}%)")
        print(f"   Desktop: ${ml_data['desktop_revenue']:.2f} ({ml_data['desktop_revenue']/total_rev*100:.1f}%)")
        
        print(f"👤 User Value Breakdown:")
        print(f"   High Value Users: ${ml_data['high_value_revenue']:.2f} ({ml_data['high_value_revenue']/total_rev*100:.1f}%)")
        print(f"   Low Value Users:  ${ml_data['low_value_revenue']:.2f} ({ml_data['low_value_revenue']/total_rev*100:.1f}%)")

    print("="*80)
    print(f"⏱️ Total Time: {duration:.2f}s")

if __name__ == "__main__":
    run_revenue_simulation()