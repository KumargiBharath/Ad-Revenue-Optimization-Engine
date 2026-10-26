from dataclasses import dataclass
from typing import Dict, Any, List
import yaml
from pathlib import Path

# ==============================
# Config Dataclasses
# ==============================

@dataclass
class PositionBiasConfig:
    top: float
    side: float
    bottom: float

@dataclass
class UserAttributesConfig:
    age_groups: List[str]
    devices: List[str]
    locations: List[str]

@dataclass
class DataGenerationConfig:
    num_impressions: int
    num_advertisers: int
    min_bid: float
    max_bid: float
    min_budget: float
    max_budget: float
    competitors_per_auction: int
    position_bias: PositionBiasConfig
    user_attributes: UserAttributesConfig

@dataclass
class ClickModelConfig:
    quality_weight: float
    match_weight: float
    historical_ctr_weight: float
    position_weight: float
    time_weight: float
    noise_std: float

@dataclass
class ModelConfig:
    type: str

@dataclass
class TrainingConfig:
    test_size: float
    random_state: int

@dataclass
class GlobalConfig:
    random_seed: int
    mode: str
    data_generation: DataGenerationConfig
    click_model: ClickModelConfig
    model: ModelConfig
    training: TrainingConfig
    hyperparameters: Dict[str, Dict[str, Any]]

    @property
    def active_hyperparameters(self) -> Dict[str, Any]:
        return self.hyperparameters.get(self.model.type, {})

# ==============================
# Config Loader
# ==============================

def load_config(project_root_path: str) -> GlobalConfig:
    """
    Loads both config.yaml and model_config.yaml and merges them.
    """
    project_root = Path(project_root_path)

    # Define paths to both config files
    sim_path = project_root / "configs" / "config.yaml"
    model_path = project_root / "configs" / "model_config.yaml"

    # 1. Load Simulation Config
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation config not found at {sim_path}")
    with open(sim_path, "r") as f:
        sim_cfg = yaml.safe_load(f)

    # 2. Load Model Config
    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found at {model_path}")
    with open(model_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # 3. Parse Simulation Details
    position_bias = PositionBiasConfig(**sim_cfg["data_generation"]["position_bias"])
    user_attributes = UserAttributesConfig(**sim_cfg["data_generation"]["user_attributes"])
    
    data_generation = DataGenerationConfig(
        num_impressions=sim_cfg["data_generation"]["num_impressions"],
        num_advertisers=sim_cfg["data_generation"]["num_advertisers"],
        min_bid=sim_cfg["data_generation"]["min_bid"],
        max_bid=sim_cfg["data_generation"]["max_bid"],
        min_budget=sim_cfg["data_generation"]["min_budget"],
        max_budget=sim_cfg["data_generation"]["max_budget"],
        competitors_per_auction=sim_cfg["data_generation"]["competitors_per_auction"],
        position_bias=position_bias,
        user_attributes=user_attributes
    )

    click_model = ClickModelConfig(**sim_cfg["click_model"])

    # 4. Parse Model Details (from model_config.yaml)
    model = ModelConfig(**model_cfg["model"])
    training = TrainingConfig(**model_cfg["training"])
    hyperparameters = model_cfg["hyperparameters"]

    # 5. Combine into GlobalConfig
    return GlobalConfig(
        random_seed=sim_cfg["random_seed"],
        mode=sim_cfg["mode"],
        data_generation=data_generation,
        click_model=click_model,
        model=model,
        training=training,
        hyperparameters=hyperparameters
    )