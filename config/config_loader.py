"""Enhanced configuration loader with market-specific config support.

This module provides hierarchical configuration loading:
1. Base strategy config (defines strategy logic)
2. Market-specific configs (override market-specific parameters)
3. Config inheritance and merging
4. Saved/optimized config profiles

Industry Standard Approach:
- Similar to Django settings (base.py + local.py)
- Similar to Kubernetes configs (base + overlays)
- Similar to NinjaTrader instrument-specific settings
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import yaml
import copy


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge override dict into base dict.
    
    Values in override take precedence. Nested dicts are merged recursively.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(config_path: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve config path (supports relative and absolute paths).
    
    Args:
        config_path: Config path (relative or absolute)
        base_dir: Base directory for relative paths (default: current working directory)
        
    Returns:
        Resolved Path object
    """
    path = Path(config_path)
    
    if path.is_absolute():
        return path
    
    if base_dir:
        return (base_dir / path).resolve()
    
    return Path.cwd() / path


def load_strategy_config_with_market(
    base_config_path: Path,
    market_symbol: Optional[str] = None,
    market_config_path: Optional[Path] = None,
    config_profile: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load strategy configuration with market-specific overrides.
    
    This function implements hierarchical config loading:
    1. Load base strategy config
    2. If market_config_path provided, load and merge market config
    3. If market_symbol provided, try to find market config automatically
    4. If config_profile provided, load saved/optimized config
    
    Args:
        base_config_path: Path to base strategy config file
        market_symbol: Trading symbol (e.g., 'EURUSD', 'MES')
        market_config_path: Optional explicit path to market config file
        config_profile: Optional config profile name (e.g., 'optimized_eurusd_2024')
        
    Returns:
        Merged configuration dictionary
        
    Example:
        # Load base config with EURUSD market config
        config = load_strategy_config_with_market(
            Path('strategies/titanmaster/config.yml'),
            market_symbol='EURUSD'
        )
        
        # Load with explicit market config path
        config = load_strategy_config_with_market(
            Path('strategies/titanmaster/config.yml'),
            market_config_path=Path('strategies/titanmaster/configs/EURUSD.yml')
        )
        
        # Load with saved optimized config
        config = load_strategy_config_with_market(
            Path('strategies/titanmaster/config.yml'),
            market_symbol='EURUSD',
            config_profile='optimized_2024'
        )
    """
    # Load base config
    base_config = load_yaml_config(base_config_path)
    base_dir = base_config_path.parent
    
    # Check if base config references a market config
    if 'market_config' in base_config:
        market_config_ref = base_config.pop('market_config')
        if isinstance(market_config_ref, str):
            # Relative path from base config directory
            ref_path = base_dir / market_config_ref
            if ref_path.exists():
                market_config_path = ref_path
        elif isinstance(market_config_ref, dict):
            # Config reference with symbol matching
            if market_symbol and market_symbol in market_config_ref:
                ref_path = base_dir / market_config_ref[market_symbol]
                if ref_path.exists():
                    market_config_path = ref_path
    
    # Load market config if path provided
    if market_config_path:
        market_config = load_yaml_config(market_config_path)
        base_config = deep_merge(base_config, market_config)
    
    # Try to auto-find market config if symbol provided
    elif market_symbol:
        # Try standard locations:
        # 1. strategies/{strategy}/configs/{symbol}.yml
        # 2. strategies/{strategy}/configs/{symbol}_config.yml
        # 3. strategies/{strategy}/configs/{market_symbol}.yml
        configs_dir = base_dir / 'configs'
        possible_paths = [
            configs_dir / f'{market_symbol}.yml',
            configs_dir / f'{market_symbol}_config.yml',
            configs_dir / f'{market_symbol.lower()}.yml',
        ]
        
        market_config_loaded = False
        for path in possible_paths:
            if path.exists():
                market_config = load_yaml_config(path)
                base_config = deep_merge(base_config, market_config)
                print(f"  ✓ Found and loaded market config: {path}")
                market_config_loaded = True
                break
        
        if not market_config_loaded:
            print(f"  ⚠ No market config found for {market_symbol} in {configs_dir}")
            print(f"     Tried: {[str(p) for p in possible_paths]}")
    
    # Load config profile if specified
    if config_profile:
        profiles_dir = base_dir / 'configs' / 'profiles'
        profile_path = profiles_dir / f'{config_profile}.yml'
        
        if profile_path.exists():
            profile_config = load_yaml_config(profile_path)
            base_config = deep_merge(base_config, profile_config)
        else:
            raise FileNotFoundError(f"Config profile not found: {profile_path}")
    
    return base_config


def save_config_profile(
    config: Dict[str, Any],
    strategy_name: str,
    profile_name: str,
    description: Optional[str] = None
) -> Path:
    """
    Save a configuration as a named profile.
    
    Useful for saving optimized configurations or market-specific settings.
    
    Args:
        config: Configuration dictionary to save
        strategy_name: Strategy name (e.g., 'titanmaster')
        profile_name: Profile name (e.g., 'optimized_eurusd_2024')
        description: Optional description for the profile
        
    Returns:
        Path to saved profile file
        
    Example:
        # Save optimized config
        save_config_profile(
            optimized_config,
            'titanmaster',
            'optimized_eurusd_2024',
            'Optimized for EURUSD 2021-2023'
        )
    """
    strategy_dir = Path('strategies') / strategy_name
    profiles_dir = strategy_dir / 'configs' / 'profiles'
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    profile_path = profiles_dir / f'{profile_name}.yml'
    
    # Add metadata
    profile_config = {
        '_profile_metadata': {
            'name': profile_name,
            'description': description or f'Config profile: {profile_name}',
            'created_at': datetime.now().isoformat(),
        },
        **config
    }
    
    with open(profile_path, 'w') as f:
        yaml.dump(profile_config, f, default_flow_style=False, sort_keys=False)
    
    return profile_path

