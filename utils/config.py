#!/usr/bin/env python3
"""
Configuration Management for TopoDiffuser

Supports:
- YAML config files
- Command-line overrides
- Config merging (base configs)
- Validation
"""

import yaml
import os
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


class ConfigDict(dict):
    """Dictionary that allows dot notation access."""
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    @classmethod
    def from_dict(cls, d):
        """Recursively convert dict to ConfigDict."""
        result = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = cls.from_dict(v)
            else:
                result[k] = v
        return result
    
    def to_dict(self):
        """Convert back to regular dict."""
        result = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def load_yaml(path: str) -> Dict:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge override into base.
    """
    result = base.copy()
    for key, value in override.items():
        if key == "base_config":
            continue  # Skip metadata
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, config_dir: str = "configs") -> ConfigDict:
    """
    Load config with support for base config inheritance.
    
    Args:
        config_path: Path to config file (relative to config_dir or absolute)
        config_dir: Directory containing configs
        
    Returns:
        ConfigDict with merged configuration
    """
    # Resolve path
    if not os.path.isabs(config_path):
        # Find config directory relative to project root
        project_root = Path(__file__).parent.parent
        config_dir = project_root / config_dir
        config_path = config_dir / config_path
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load primary config
    config = load_yaml(config_path)
    
    # Handle base config inheritance
    if "base_config" in config:
        base_path = config["base_config"]
        if not os.path.isabs(base_path):
            base_path = config_path.parent / base_path
        
        if base_path.exists():
            base_config = load_yaml(base_path)
            # Merge primary over base
            config = merge_configs(base_config, config)
        else:
            print(f"Warning: Base config not found: {base_path}")
    
    return ConfigDict.from_dict(config)


def override_config(config: ConfigDict, overrides: Dict[str, Any]) -> ConfigDict:
    """
    Override config values using dot notation keys.
    
    Example:
        overrides = {"training.batch_size": 64, "model.encoder.input_channels": 5}
    """
    config = ConfigDict.from_dict(config.to_dict())  # Deep copy
    
    for key, value in overrides.items():
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = value
    
    return config


def get_default_config() -> ConfigDict:
    """Get default configuration."""
    project_root = Path(__file__).parent.parent
    default_config_path = project_root / "configs" / "default.yaml"
    return load_config(str(default_config_path))


# =============================================================================
# Typed Config Classes (for IDE support and validation)
# =============================================================================

@dataclass
class DataConfig:
    """Data configuration."""
    train_sequences: list = field(default_factory=lambda: ["00", "02", "05", "07"])
    val_sequences: list = field(default_factory=lambda: ["08", "09", "10"])
    test_sequences: list = field(default_factory=lambda: ["08", "09", "10"])
    bev_height: int = 300
    bev_width: int = 400
    num_future: int = 8
    waypoint_spacing: float = 2.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    encoder_input_channels: int = 3
    conditioning_dim: int = 512
    diffusion_timesteps: int = 10
    denoiser_arch: str = "mlp"


@dataclass
class TrainingConfig:
    """Training configuration."""
    mode: str = "joint"  # "joint" or "separate"
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    alpha_road: float = 0.1
    mixed_precision: bool = True


@dataclass
class Config:
    """Top-level configuration."""
    project: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# =============================================================================
# Validation
# =============================================================================

def validate_config(config: ConfigDict) -> bool:
    """
    Validate configuration.
    
    Returns True if valid, raises ValueError otherwise.
    """
    required_sections = ["project", "paths", "data", "model", "training"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate paths exist
    data_root = config.paths.get("data_root")
    if data_root and not os.path.exists(data_root):
        print(f"Warning: Data root does not exist: {data_root}")
    
    # Validate model config
    if config.model.encoder.input_channels not in [3, 5]:
        print(f"Warning: Unusual input channels: {config.model.encoder.input_channels}")
    
    # Validate training config
    if config.training.batch_size < 1:
        raise ValueError("Batch size must be >= 1")
    
    if not 0 <= config.training.loss.alpha_road <= 10:
        print(f"Warning: alpha_road is {config.training.loss.alpha_road}, typical range is 0-1")
    
    return True


# =============================================================================
# CLI Integration
# =============================================================================

def config_from_args(args) -> ConfigDict:
    """
    Build config from argparse args.
    
    Expected args:
        - config: Path to config file
        - overrides: List of key=value pairs for overrides
    """
    # Load base config
    if hasattr(args, 'config') and args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Apply overrides
    if hasattr(args, 'overrides') and args.overrides:
        overrides = {}
        for override in args.overrides:
            if '=' in override:
                key, value = override.split('=', 1)
                # Try to parse value as number/bool
                try:
                    value = eval(value)
                except:
                    pass  # Keep as string
                overrides[key] = value
        config = override_config(config, overrides)
    
    # Validate
    validate_config(config)
    
    return config


if __name__ == "__main__":
    # Test config loading
    print("Testing Configuration System...")
    print("=" * 60)
    
    # Test 1: Load default
    print("\n1. Loading default config...")
    default = get_default_config()
    print(f"   Project: {default.project.name}")
    print(f"   Batch size: {default.training.batch_size}")
    print(f"   Alpha road: {default.training.loss.alpha_road}")
    
    # Test 2: Load specific config
    print("\n2. Loading lidar_only config...")
    lidar_config = load_config("lidar_only.yaml")
    print(f"   Project: {lidar_config.project.name}")
    print(f"   Input channels: {lidar_config.model.encoder.input_channels}")
    
    # Test 3: Override
    print("\n3. Testing overrides...")
    overrides = {"training.batch_size": 128, "training.loss.alpha_road": 0.5}
    modified = override_config(default, overrides)
    print(f"   New batch size: {modified.training.batch_size}")
    print(f"   New alpha: {modified.training.loss.alpha_road}")
    
    # Test 4: Validation
    print("\n4. Testing validation...")
    try:
        validate_config(default)
        print("   ✓ Config is valid")
    except ValueError as e:
        print(f"   ✗ Validation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Config system working!")
