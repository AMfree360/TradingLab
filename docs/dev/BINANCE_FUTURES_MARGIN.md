# Binance Futures Margin & Leverage - Industry Standards

## Overview

This document explains how margin and leverage work for Binance futures, compares it to industry standards, and describes how to properly implement it in the backtest engine.

## Binance Futures Margin System

### How Binance Futures Works

Binance perpetual futures use a **leverage-based margin system**, not fixed margin per contract like traditional futures exchanges (CME, ICE).

#### Key Concepts

1. **Leverage**
   - Binance offers up to **125x leverage** for BTCUSDT
   - Leverage is set per position (not per account)
   - Higher leverage = lower margin requirement

2. **Initial Margin**
   - Calculated as: `Initial Margin = Notional Value / Leverage`
   - Where: `Notional Value = Contract Price × Contract Quantity`
   - Example: 1 BTC at $50,000 with 10x leverage = $5,000 initial margin

3. **Maintenance Margin**
   - Lower threshold than initial margin (typically 0.5-1% of notional)
   - If equity falls below maintenance margin, position is liquidated
   - Formula: `Maintenance Margin = Notional Value × Maintenance Margin Rate`
   - Example: 1 BTC at $50,000 with 0.5% rate = $250 maintenance margin

4. **Margin Modes**

   **Isolated Margin** (Position-specific):
   - Margin allocated to a single position
   - If liquidated, only that position's margin is lost
   - Safer for risk management
   - Formula: `Isolated Margin = Notional Value / Leverage`

   **Cross Margin** (Account-wide):
   - Margin shared across all positions
   - Entire account balance can be used to cover losses
   - More capital efficient but riskier
   - Formula: `Available Margin = Total Equity - Total Used Margin`

5. **Liquidation**
   - Occurs when: `Equity < Maintenance Margin`
   - Binance uses a **liquidation price** calculation
   - Partial liquidation possible (reduce position size instead of full close)

### Binance Margin Calculation Example

**Scenario**: Trading 1 BTCUSDT contract at $50,000 with 10x leverage

```
Notional Value = $50,000 × 1 = $50,000
Initial Margin = $50,000 / 10 = $5,000
Maintenance Margin = $50,000 × 0.005 = $250 (0.5% rate)

With $10,000 account:
- Can open: 2 contracts (2 × $5,000 = $10,000)
- Free margin: $0
- Liquidation if equity < $250 per contract
```

## Industry Standards Comparison

### Traditional Futures (CME, ICE, etc.)

**Fixed Margin System**:
- Exchange sets fixed margin per contract
- Margin doesn't change with price (until exchange adjusts)
- Example: MES (Micro E-mini S&P 500)
  - Initial margin: $2,219 per contract (fixed)
  - Intraday margin: $50 per contract (fixed)
  - Margin is NOT calculated from leverage

**Current Engine Implementation**: ✅ Supports this via `initial_margin_per_contract` and `intraday_margin_per_contract`

### Crypto Futures (Binance, Bybit, etc.)

**Leverage-Based Margin System**:
- Margin calculated from leverage: `Margin = Notional / Leverage`
- Maintenance margin rate (typically 0.5-1%)
- Leverage can be changed per position
- Example: BTCUSDT at $50k with 10x leverage
  - Initial margin: $5,000 (calculated)
  - Maintenance margin: $250 (0.5% of notional)

**Current Engine Implementation**: ⚠️ Partially supports (falls back to leverage-based if fixed margin not set)

### Forex (Spot & CFDs)

**Leverage-Based Margin System**:
- Similar to crypto futures
- Margin = Notional / Leverage
- Example: EURUSD at 1.1000, 1 lot (100,000 units), 30x leverage
  - Notional: 1.1000 × 100,000 = $110,000
  - Margin: $110,000 / 30 = $3,666.67

**Current Engine Implementation**: ✅ Fully supports

## Current Engine Implementation

### What Works Well

1. **Leverage-Based Margin** (for non-futures):
   ```python
   # engine/market.py
   margin = (entry_price * quantity) / leverage
   ```
   ✅ Correct for forex, crypto spot, stocks

2. **Fixed Margin** (for traditional futures):
   ```python
   # engine/market.py
   if asset_class == 'futures':
       margin = quantity * initial_margin_per_contract
   ```
   ✅ Correct for CME-style futures (MES, etc.)

3. **P&L Independence**:
   ```python
   # P&L is always: (price_change) * quantity
   # Leverage does NOT affect P&L
   ```
   ✅ Industry standard

### What Needs Enhancement

1. **Binance-Style Futures Support**:
   - Currently: Falls back to leverage-based if `initial_margin_per_contract` not set
   - Problem: No maintenance margin calculation
   - Problem: No isolated vs cross margin mode
   - Problem: No leverage per position (uses account-level leverage)

2. **Maintenance Margin**:
   - Currently: Only checks `equity < margin_used * margin_call_level`
   - Needed: Separate maintenance margin calculation
   - Needed: Position-level liquidation checks

3. **Margin Mode**:
   - Currently: Always uses cross margin (account-wide)
   - Needed: Support isolated margin mode

## Recommended Implementation

### Option 1: Enhance MarketSpec for Binance Futures (Recommended)

Add new fields to `MarketSpec`:

```python
@dataclass
class MarketSpec:
    # ... existing fields ...
    
    # For Binance-style futures (leverage-based)
    margin_mode: Literal["isolated", "cross"] = "cross"
    maintenance_margin_rate: Optional[float] = None  # e.g., 0.005 for 0.5%
    
    # For traditional futures (fixed margin)
    initial_margin_per_contract: Optional[float] = None
    intraday_margin_per_contract: Optional[float] = None
```

### Option 2: Separate Margin Calculation Methods

Enhance `calculate_margin()` to support both systems:

```python
def calculate_margin(
    self, 
    entry_price: float, 
    quantity: float, 
    is_intraday: bool = True,
    leverage: Optional[float] = None  # Position-level leverage
) -> float:
    """Calculate margin required for a position.
    
    Supports two systems:
    1. Traditional futures: Fixed margin per contract
    2. Binance-style futures: Leverage-based margin
    
    Args:
        entry_price: Entry price
        quantity: Position quantity
        is_intraday: True for intraday/day trading
        leverage: Position-level leverage (overrides market_spec.leverage)
    
    Returns:
        Margin required in cash units
    """
    # Use position leverage if provided, otherwise market leverage
    effective_leverage = leverage or self.leverage
    
    if self.asset_class == 'futures':
        # Check if using fixed margin (traditional) or leverage-based (Binance)
        if self.initial_margin_per_contract is not None:
            # Traditional futures: Fixed margin
            if is_intraday and self.intraday_margin_per_contract is not None:
                return abs(quantity) * self.intraday_margin_per_contract
            return abs(quantity) * self.initial_margin_per_contract
        else:
            # Binance-style futures: Leverage-based margin
            notional_value = entry_price * quantity
            return notional_value / effective_leverage
    
    # Other asset classes: Leverage-based
    notional_value = entry_price * quantity
    return notional_value / effective_leverage
```

### Option 3: Add Maintenance Margin Calculation

```python
def calculate_maintenance_margin(
    self,
    entry_price: float,
    quantity: float,
    current_price: Optional[float] = None
) -> float:
    """Calculate maintenance margin required.
    
    For Binance-style futures: Uses maintenance_margin_rate
    For traditional futures: Typically same as initial margin (or lower)
    
    Args:
        entry_price: Entry price
        quantity: Position quantity
        current_price: Current price (for mark-to-market, uses entry if None)
    
    Returns:
        Maintenance margin required
    """
    price = current_price or entry_price
    notional_value = price * quantity
    
    if self.asset_class == 'futures':
        if self.maintenance_margin_rate is not None:
            # Binance-style: Percentage of notional
            return notional_value * self.maintenance_margin_rate
        elif self.initial_margin_per_contract is not None:
            # Traditional: Use initial margin (or could be lower)
            return abs(quantity) * self.initial_margin_per_contract
        else:
            # Fallback: Use initial margin calculation
            return self.calculate_margin(entry_price, quantity)
    
    # For other assets, maintenance = initial margin
    return self.calculate_margin(entry_price, quantity)
```

### Option 4: Enhanced Liquidation Check

```python
def check_liquidation(
    self,
    entry_price: float,
    current_price: float,
    quantity: float,
    account_equity: float,
    margin_mode: str = "cross"
) -> bool:
    """Check if position should be liquidated.
    
    Liquidation occurs when:
    - Isolated: Position equity < Position maintenance margin
    - Cross: Account equity < Total maintenance margin
    
    Args:
        entry_price: Entry price
        current_price: Current market price
        quantity: Position quantity
        account_equity: Total account equity
        margin_mode: "isolated" or "cross"
    
    Returns:
        True if position should be liquidated
    """
    # Calculate position P&L
    if quantity > 0:  # Long
        position_pnl = (current_price - entry_price) * quantity
    else:  # Short
        position_pnl = (entry_price - current_price) * abs(quantity)
    
    # Calculate maintenance margin
    maintenance_margin = self.calculate_maintenance_margin(
        entry_price, quantity, current_price
    )
    
    if margin_mode == "isolated":
        # Isolated: Check position-level equity
        position_equity = maintenance_margin + position_pnl
        return position_equity < maintenance_margin
    else:
        # Cross: Check account-level equity
        return account_equity < maintenance_margin
```

## Market Profile Configuration

### Binance BTCUSDT Futures (Leverage-Based)

```yaml
BTCUSDT_FUTURES:
  exchange: binance
  symbol: BTCUSDT
  asset_class: futures
  market_type: futures
  leverage: 10.0  # Default leverage (can be overridden per position)
  margin_mode: cross  # or "isolated"
  maintenance_margin_rate: 0.005  # 0.5% maintenance margin
  contract_size: 1.0
  tick_value: 0.01
  commission_per_contract: 2.0
  commission_rate: 0.0
  # Note: initial_margin_per_contract is NOT set (uses leverage-based)
```

### Traditional Futures (Fixed Margin)

```yaml
MES:
  exchange: cme
  symbol: MES
  asset_class: futures
  market_type: futures
  initial_margin_per_contract: 2219.0  # Fixed margin
  intraday_margin_per_contract: 50.0
  contract_size: 5.0
  tick_value: 1.25
  commission_per_contract: 0.87
  # Note: leverage is ignored (uses fixed margin)
```

## Implementation Priority

### Phase 1: Basic Binance Support (High Priority)
1. ✅ Add `maintenance_margin_rate` to MarketSpec
2. ✅ Enhance `calculate_margin()` to detect leverage-based vs fixed
3. ✅ Update market_profiles.yml with Binance futures config

### Phase 2: Advanced Features (Medium Priority)
1. ⚠️ Add `margin_mode` (isolated vs cross)
2. ⚠️ Implement position-level liquidation checks
3. ⚠️ Support position-level leverage (override market leverage)

### Phase 3: Advanced Risk Management (Low Priority)
1. ⚠️ Partial liquidation support
2. ⚠️ Funding rate simulation (for perpetuals)
3. ⚠️ Auto-deleveraging (ADL) simulation

## Testing

### Test Case 1: Binance BTCUSDT with 10x Leverage

```python
# Setup
market_spec = MarketSpec(
    symbol="BTCUSDT",
    asset_class="futures",
    leverage=10.0,
    maintenance_margin_rate=0.005,
    # initial_margin_per_contract NOT set (uses leverage-based)
)

# Test
entry_price = 50000.0
quantity = 1.0

# Initial margin
initial_margin = market_spec.calculate_margin(entry_price, quantity)
assert initial_margin == 5000.0  # $50,000 / 10 = $5,000

# Maintenance margin
maintenance_margin = market_spec.calculate_maintenance_margin(entry_price, quantity)
assert maintenance_margin == 250.0  # $50,000 × 0.005 = $250
```

### Test Case 2: Traditional Futures (MES)

```python
# Setup
market_spec = MarketSpec(
    symbol="MES",
    asset_class="futures",
    initial_margin_per_contract=2219.0,
    intraday_margin_per_contract=50.0,
)

# Test
entry_price = 4500.0
quantity = 1.0

# Initial margin (overnight)
initial_margin = market_spec.calculate_margin(entry_price, quantity, is_intraday=False)
assert initial_margin == 2219.0  # Fixed margin

# Intraday margin
intraday_margin = market_spec.calculate_margin(entry_price, quantity, is_intraday=True)
assert intraday_margin == 50.0  # Fixed margin
```

## Summary

### Key Differences

| Feature | Traditional Futures | Binance Futures |
|---------|-------------------|-----------------|
| **Margin System** | Fixed per contract | Leverage-based |
| **Margin Calculation** | `quantity × fixed_margin` | `notional / leverage` |
| **Maintenance Margin** | Same as initial (or fixed) | Percentage of notional |
| **Leverage** | Not used | Core parameter |
| **Margin Mode** | N/A | Isolated or Cross |

### Current Status

✅ **Works**: Traditional futures (fixed margin), Forex (leverage-based), Crypto spot  
⚠️ **Partial**: Binance futures (leverage works, but missing maintenance margin)  
❌ **Missing**: Isolated margin mode, position-level leverage, maintenance margin checks

### Recommendation

Implement **Phase 1** enhancements to properly support Binance-style futures while maintaining backward compatibility with traditional futures.

