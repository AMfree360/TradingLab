# Hierarchical Configuration System

## Problem Statement

When testing strategies across different markets/instruments:
- Optimized parameters for one market conflict with another
- Switching markets requires manually editing config files
- Risk of overwriting optimized settings
- No way to save and reuse optimized configurations

## Solution: Hierarchical Config System

### Architecture

```
Base Config (Strategy Logic)
    ↓
Market Config (Market-Specific Overrides)
    ↓
Config Profile (Optimized Settings - Optional)
    ↓
Final Merged Config
```

### Key Features

1. **Base Config**: Contains strategy logic (timeframes, indicators, filters)
2. **Market Configs**: Override market-specific parameters (stop distances, commissions)
3. **Config Profiles**: Save optimized configs for easy reuse
4. **Auto-Discovery**: Automatically finds market configs based on symbol
5. **Deep Merging**: Recursively merges nested dictionaries

## Implementation

### Files Created

1. **`config/config_loader.py`**: Hierarchical config loading logic
2. **`strategies/ema_crossover/configs/EURUSD.yml`**: Example EURUSD config
3. **`strategies/ema_crossover/configs/MES.yml`**: Example MES config
4. **`strategies/ema_crossover/CONFIG_GUIDE.md`**: User guide

### Usage Examples

#### Auto-Discovery (Recommended)

```bash
# Automatically loads configs/EURUSD.yml
python3 scripts/run_backtest.py \
    --strategy ema_crossover \
  --market EURUSD \
  --data data/raw/EURUSD_M15_2021_2025.csv
```

#### With Saved Profile

```bash
python3 scripts/run_backtest.py \
    --strategy ema_crossover \
  --market EURUSD \
  --config-profile optimized_eurusd_2024 \
  --data data/raw/EURUSD_M15_2021_2025.csv
```

#### Programmatic Usage

```python
from config.config_loader import load_strategy_config_with_market, save_config_profile
from pathlib import Path

# Load with market config
config = load_strategy_config_with_market(
    Path('strategies/ema_crossover/config.yml'),
    market_symbol='EURUSD'
)

# Save optimized config
save_config_profile(
    optimized_config,
    'ema_crossover',
    'optimized_eurusd_2024',
    'Optimized for EURUSD 2021-2023'
)
```

## Industry Standards Comparison

| System | Approach | Our Implementation |
|--------|----------|-------------------|
| **Django** | `base.py` + `local.py` | Base config + Market configs |
| **Kubernetes** | Base + Overlays | Base config + Market configs + Profiles |
| **NinjaTrader** | Instrument-specific settings | Market configs per symbol |
| **MT5** | Symbol-specific parameters | Market configs per symbol |

## Benefits

1. ✅ **Separation of Concerns**: Strategy logic separate from market params
2. ✅ **No Conflicts**: Each market has its own optimized settings
3. ✅ **Easy Switching**: Just change `--market` flag
4. ✅ **Version Control**: Market configs can be tracked separately
5. ✅ **Reusability**: Save optimized configs as profiles
6. ✅ **Backward Compatible**: Single config.yml still works

## Directory Structure

```
strategies/ema_crossover/
├── config.yml                    # Base strategy config
├── strategy.py                   # Strategy implementation
├── configs/
│   ├── EURUSD.yml               # EURUSD market config
│   ├── MES.yml                  # MES futures config
│   ├── BTCUSDT.yml              # BTCUSDT crypto config
│   └── profiles/
│       ├── optimized_eurusd_2024.yml
│       └── optimized_mes_2024.yml
└── CONFIG_GUIDE.md              # User documentation
```

## Migration Path

### For Existing Users

1. **No changes required** - single config.yml still works
2. **Optional**: Extract market-specific values to `configs/{SYMBOL}.yml`
3. **Optional**: Use `--market` flag for auto-discovery

### For New Users

1. Create base config with market-agnostic values
2. Create market configs for each instrument
3. Use `--market` flag when running backtests

## Future Enhancements

1. **Config Templates**: Pre-built configs for common markets
2. **Config Validation**: Ensure market configs are compatible
3. **Config Diff**: Compare configs to see what changed
4. **Config Import/Export**: Share configs between strategies
5. **Config Versioning**: Track config changes over time

