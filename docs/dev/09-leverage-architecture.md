# Leverage Architecture - Industry Standard Implementation

## Overview

This document describes the new leverage architecture that follows industry-standard separation of concerns. **Leverage belongs in the engine/broker layer, not in strategies.**

## Key Principles

### 1. Separation of Concerns

```
Strategy Layer (Market-Agnostic)
  └─ Emits trade intent: direction, stop, target, risk_pct
      ↓
Execution Engine
  └─ Requests position sizing
      ↓
Broker / MarketSpec Layer
  ├─ Margin rules
  ├─ Leverage
  ├─ Contract specs
  └─ Commission model
      ↓
Account
  └─ Cash, margin, equity
```

### 2. Leverage Only Affects Margin, NOT P&L

**Critical Rule**: Leverage controls how much margin you must post, but it does NOT change profits or losses.

- ✅ **Correct**: Margin = (entry_price × quantity) / leverage
- ✅ **Correct**: P&L = (price_change) × quantity (independent of leverage)
- ❌ **Wrong**: P&L = (price_change) × quantity × leverage

### 3. Strategies Are Market-Agnostic

Strategies should never know about:
- Leverage
- Margin requirements
- Contract sizes
- Market-specific rules

Strategies only specify:
- Direction (long/short)
- Entry price
- Stop price
- Risk percentage

## Architecture Components

### MarketSpec Class

Located in `engine/market.py`, this class encapsulates all market-specific rules:

```python
@dataclass
class MarketSpec:
    symbol: str
    exchange: str
    asset_class: Literal["forex", "crypto", "stock", "futures"]
    leverage: float  # Leverage multiplier (1.0 = no leverage)
    contract_size: Optional[float]  # For forex/futures
    pip_value: Optional[float]  # For forex
    commission_rate: float
    slippage_ticks: float
    # ... other market-specific parameters
```

**Key Methods**:
- `calculate_margin(entry_price, quantity)` → Returns margin required
- `calculate_unrealized_pnl(...)` → Returns P&L (leverage-independent)
- `load_from_profiles(symbol)` → Loads from `config/market_profiles.yml`

### Broker Class

Located in `engine/broker.py` (`BrokerModel`), this class handles margin and position validation:

```python
@dataclass
class BrokerModel:
    market_spec: MarketSpec
    margin_call_level: float = 1.0
    
    def calculate_margin_required(...) → float
    def can_afford_position(...) → (bool, required_cash)
    def check_margin_call(equity, margin_used) → bool
    def adjust_quantity_for_cash(...) → float
```

## How It Works

### 1. Market Profile Loading

Market profiles are defined in `config/market_profiles.yml`:

```yaml
markets:
  EURUSD:
    exchange: oanda
    symbol: EURUSD
    asset_class: forex
    leverage: 30.0  # 30:1 leverage
    commission_rate: 0.0
    slippage_ticks: 0.5
    contract_size: 100000
    pip_value: 0.0001
```

### 2. Engine Initialization

The backtest engine automatically loads MarketSpec from profiles:

```python
engine = BacktestEngine(
    strategy=strategy,
    market_spec=None,  # Auto-loads from market profiles
    initial_capital=10000.0
)
```

If MarketSpec is not provided, the engine:
1. Tries to load from `market_profiles.yml` using strategy's symbol
2. Falls back to strategy config (backward compatibility)

### 3. Position Sizing

Position sizing uses Broker methods:

```python
# Calculate quantity based on risk (strategy intent)
quantity = risk_amount / price_risk

# Check if we can afford it (broker constraint)
can_afford, required_cash = broker.can_afford_position(
    entry_price, quantity, available_cash
)

# Adjust if needed
if not can_afford:
    quantity = broker.adjust_quantity_for_cash(...)
```

### 4. Margin Calculation

Margin is calculated using MarketSpec:

```python
margin = market_spec.calculate_margin(entry_price, quantity)
# Equivalent to: (entry_price * quantity) / leverage
```

### 5. P&L Calculation

P&L is calculated independently of leverage:

```python
# Long position
unrealized_pnl = (current_price - entry_price) * quantity

# Short position
unrealized_pnl = (entry_price - current_price) * quantity
```

**Note**: Leverage does NOT appear in P&L calculations.

## Benefits

### 1. Strategy Portability

The same strategy can be tested on different markets:

```bash
# EURUSD with 30:1 leverage
python3 scripts/run_backtest.py --strategy ema_crossover --market EURUSD

# BTCUSDT with 1:1 leverage (spot)
python3 scripts/run_backtest.py --strategy ema_crossover --market BTCUSDT

# ES futures with exchange-set margin
python3 scripts/run_backtest.py --strategy ema_crossover --market ES
```

### 2. Accurate Margin Modeling

- Margin requirements are correctly calculated per market
- Margin calls are enforced realistically
- Position sizing respects margin constraints

### 3. Clean Separation

- Strategies don't need market-specific code
- Market rules are centralized in profiles
- Easy to add new markets without touching strategies

## Migration Guide

### For Strategy Developers

**Before** (❌ Wrong):
```yaml
# strategies/my_strategy/config.yml
market:
  leverage: 100.0  # Don't do this!
```

**After** (✅ Correct):
```yaml
# strategies/my_strategy/config.yml
market:
  symbol: EURUSD  # Only specify symbol
  # Leverage comes from market_profiles.yml
```

### For Market Profile Updates

Update `config/market_profiles.yml`:

```yaml
markets:
  EURUSD:
    leverage: 30.0  # Set leverage here
    # ... other market settings
```

### Backward Compatibility

The engine maintains backward compatibility:
- If MarketSpec is not found in profiles, it falls back to strategy config
- Existing strategies continue to work
- Leverage in strategy configs is deprecated but still supported

## Example: EURUSD with 30:1 Leverage

### Market Profile
```yaml
EURUSD:
  leverage: 30.0
  contract_size: 100000
  pip_value: 0.0001
```

### Position Entry
```python
# Strategy wants to risk 0.5% = $50
# Stop is 25 pips = 0.0025
quantity = $50 / 0.0025 = 20,000 units

# Broker calculates margin
margin = (1.1000 * 20,000) / 30.0 = $733.33

# Cash deducted: $733.33 (not $22,000!)
```

### P&L Calculation
```python
# Price moves 10 pips (0.0010)
pnl = 0.0010 * 20,000 = $20

# Leverage does NOT affect P&L
# Whether leverage is 1:1 or 100:1, P&L is still $20
```

## Testing

To verify the architecture:

1. **Test same strategy on different markets**:
  ```bash
  python3 scripts/run_backtest.py --strategy ema_crossover --market EURUSD
  python3 scripts/run_backtest.py --strategy ema_crossover --market BTCUSDT
  ```

2. **Test leverage changes**:
  ```bash
  # With 30:1 leverage
  python3 scripts/run_backtest.py --strategy ema_crossover --market EURUSD --leverage 30.0
   
  # With 1:1 leverage (no leverage)
  python3 scripts/run_backtest.py --strategy ema_crossover --market EURUSD --leverage 1.0
  ```

3. **Verify margin calls work**:
   - Run a strategy with high leverage
   - Ensure margin calls trigger when equity < margin_used

## Summary

✅ **Leverage is in the engine/broker layer** (not strategies)
✅ **P&L is independent of leverage** (only margin is affected)
✅ **Strategies are market-agnostic** (portable across markets)
✅ **Market rules are centralized** (in market_profiles.yml)
✅ **Backward compatible** (existing strategies still work)

This architecture aligns with industry standards (Backtrader, QuantConnect, VectorBT) and ensures accurate, realistic backtesting results.

