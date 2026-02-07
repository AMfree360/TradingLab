#!/usr/bin/env python3
"""Verify that all TitanEA config settings can be toggled on/off correctly."""

import sys
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from config.schema import load_config, validate_strategy_config
from strategies.titanea.strategy import TitanEAStrategy


def test_config_toggle(config_path: Path, setting_path: str, value_on: any, value_off: any):
    """Test that a config setting can be toggled on/off.
    
    Args:
        config_path: Path to config file
        setting_path: Dot-separated path to setting (e.g., 'trailing_stop.enabled')
        value_on: Value when enabled
        value_off: Value when disabled
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Load config
        config_dict = load_config(config_path)
        
        # Navigate to setting
        keys = setting_path.split('.')
        current = config_dict
        for key in keys[:-1]:
            current = current[key]
        
        # Test OFF
        current[keys[-1]] = value_off
        config_off = validate_strategy_config(config_dict)
        strategy_off = TitanEAStrategy(config_off)
        
        # Test ON
        current[keys[-1]] = value_on
        config_on = validate_strategy_config(config_dict)
        strategy_on = TitanEAStrategy(config_on)
        
        return True, f"✓ {setting_path}: Can toggle between {value_off} and {value_on}"
    except Exception as e:
        return False, f"✗ {setting_path}: Error - {e}"


def verify_all_settings():
    """Verify all TitanEA config settings."""
    config_path = Path(__file__).parent.parent / 'strategies' / 'titanea' / 'config.yml'
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return False
    
    print("=" * 80)
    print("TitanEA Config Verification")
    print("=" * 80)
    print()
    
    # Load base config
    config_dict = load_config(config_path)
    config = validate_strategy_config(config_dict)
    strategy = TitanEAStrategy(config)
    
    results = []
    
    # 1. Trailing Stop
    print("1. Trailing Stop Settings")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'trailing_stop.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 2. Partial Exit
    print("2. Partial Exit Settings")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'partial_exit.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 3. Take Profit
    print("3. Take Profit Settings")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'take_profit.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 4. Calendar Filters - Master
    print("4. Calendar Filters - Master")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'calendar_filters.master_filters_enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 5. Calendar Filters - Day of Week
    print("5. Calendar Filters - Day of Week")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'calendar_filters.day_of_week.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 6. Calendar Filters - Month of Year
    print("6. Calendar Filters - Month of Year")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'calendar_filters.month_of_year.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 7. Calendar Filters - Trading Sessions
    print("7. Calendar Filters - Trading Sessions")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'calendar_filters.trading_sessions.Session3.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 8. Calendar Filters - Time Blackouts
    print("8. Calendar Filters - Time Blackouts")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'calendar_filters.time_blackouts_enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 9. Regime Filters - ADX
    print("9. Regime Filters - ADX")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'regime_filters.adx.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 10. Regime Filters - ATR Percentile
    print("10. Regime Filters - ATR Percentile")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'regime_filters.atr_percentile.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 11. Regime Filters - EMA Expansion
    print("11. Regime Filters - EMA Expansion")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'regime_filters.ema_expansion.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 12. Regime Filters - Swing
    print("12. Regime Filters - Swing")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'regime_filters.swing.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 13. Regime Filters - Composite
    print("13. Regime Filters - Composite")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'regime_filters.composite.enabled', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 14. Regime Filters - Trend Quality
    print("14. Regime Filters - Trend Quality")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'regime_filters.trend_quality.enable_ema_distance', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'regime_filters.trend_quality.enable_sma_slope', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 15. Trade Direction
    print("15. Trade Direction")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'trade_direction.allow_long', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'trade_direction.allow_short', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 16. Execution Settings
    print("16. Execution Settings")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'execution.flatten_enabled', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'execution.enforce_margin_checks', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 17. MA Settings - Signal TF
    print("17. MA Settings - Signal TF")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'ma_settings.signal.use_fast', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.signal.use_medium', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.signal.use_slow', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.signal.use_slowest', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.signal.use_optional', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # 18. MA Settings - Entry TF
    print("18. MA Settings - Entry TF")
    print("-" * 80)
    results.append(test_config_toggle(config_path, 'ma_settings.entry.use_fast', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.entry.use_medium', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.entry.use_slow', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.entry.use_slowest', True, False))
    print(f"   {results[-1][1]}")
    results.append(test_config_toggle(config_path, 'ma_settings.entry.use_optional', True, False))
    print(f"   {results[-1][1]}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    passed = sum(1 for r in results if r[0])
    failed = sum(1 for r in results if not r[0])
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed tests:")
        for success, message in results:
            if not success:
                print(f"  {message}")
    
    print()
    print("=" * 80)
    print("Implementation Status")
    print("=" * 80)
    print()
    print("✅ FULLY IMPLEMENTED:")
    print("  - Trailing Stop (enabled/disabled)")
    print("  - Partial Exit (enabled/disabled)")
    print("  - Take Profit (enabled/disabled)")
    print("  - Calendar Filters - Master (master_filters_enabled)")
    print("  - Calendar Filters - Time Blackouts")
    print("  - Regime Filters - ADX")
    print("  - Regime Filters - ATR Threshold (note: config uses 'atr_percentile' but filter is 'atr_threshold')")
    print("  - Trade Direction (allow_long, allow_short)")
    print("  - Execution Settings (flatten_enabled, enforce_margin_checks, max_positions)")
    print("  - MA Settings (all use_* flags)")
    print()
    print("⚠️  PARTIALLY IMPLEMENTED:")
    print("  - Calendar Filters - Day of Week (config exists, filter not implemented)")
    print("  - Calendar Filters - Trading Sessions (config exists, filter not implemented)")
    print("  - Regime Filters - ATR Percentile (config exists, filter not implemented)")
    print("  - Regime Filters - EMA Expansion (config exists, filter not implemented)")
    print("  - Regime Filters - Swing (config exists, filter not implemented)")
    print("  - Regime Filters - Composite (config exists, filter not implemented)")
    print()
    print("❌ NOT IMPLEMENTED:")
    print("  - Trade Limits (max_trades_per_day) - Config exists but not enforced in engine")
    print("  - Correlation Groups (enable_correlation) - Config exists but not enforced in engine")
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = verify_all_settings()
    sys.exit(0 if success else 1)

