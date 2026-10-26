from typing import List
from .entities import Advertiser, Ad

class BudgetManager:
    def __init__(self, advertisers: List[Advertiser]):
        self.advertiser_budget = {adv.advertiser_id: adv.campaign_budget for adv in advertisers}
        self.advertiser_active = {adv.advertiser_id: True for adv in advertisers}

    def is_active(self, advertiser_id: int) -> bool:
        return self.advertiser_active.get(advertiser_id, False)

    def deduct_budget(self, ad: Ad, amount: float) -> bool:
        adv_id = ad.advertiser_id
        if not self.is_active(adv_id):
            return False

        if self.advertiser_budget[adv_id] >= amount:
            self.advertiser_budget[adv_id] -= amount
            return True
        else:
            # Deactivate if can't afford
            self.advertiser_active[adv_id] = False
            return False

    def get_active_advertisers(self) -> set: # CHANGE return type to set
        """Returns a SET of active advertiser IDs for O(1) lookup."""
        return {adv_id for adv_id, active in self.advertiser_active.items() if active}
    
    # --- FEATURES FOR PHASE 13 ISOLATION ---
    
    def get_state(self) -> dict:
        """Creates a deep copy snapshot of current budgets and active states."""
        return {
            "budget": self.advertiser_budget.copy(),
            "active": self.advertiser_active.copy()
        }

    def set_state(self, state: dict) -> None:
        """Restores budget manager to a previous snapshot."""
        self.advertiser_budget = state["budget"].copy()
        self.advertiser_active = state["active"].copy()