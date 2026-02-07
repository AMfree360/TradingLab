# Market-Agnostic Strategy Architecture

## Overview

Trading Lab is designed to be **market-agnostic**, meaning the same strategy can be tested across different markets (crypto, forex, stocks) without code changes. This is achieved through:

1. **Strategy Config**: Contains only strategy logic (indicators, signals, risk rules)
2. **Market Profiles**: Define market-specific settings (commissions, slippage, precision)
3. **Automatic Profile Loading**: System automatically applies market settings based on symbol

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Strategy Configuration                      │
│  (Strategy logic: indicators, signals, risk rules)      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Market Profile System                       │
│  (Market-specific: commissions, slippage, precision)   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Merged Configuration                        │
│  (Strategy logic + Market settings = Ready to backtest) │
└─────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Strategy Configuration

Your strategy config (`strategies/ema_crossover/config.yml`) contains:
- **Strategy logic**: Indicators, signal generation rules, risk management
- **Market placeholder**: Basic market info (can be overridden)

Example:
```yaml
strategy_name: titanea
market:
  exchange: binance
  symbol: BTCUSDT  # This can be overridden
  base_timeframe: 15m
```

### 2. Market Profiles

Market-specific settings are defined in `config/market_profiles.yml`:

```yaml
markets:
  BTCUSDT:
    commission_rate: 0.0004  # 0.04% for Binance
    slippage_ticks: 0.0
    price_precision: 2
    
  EURUSD:
    commission_rate: 0.0  # Forex uses spreads
    slippage_ticks: 0.5  # Typical forex slippage
    price_precision: 5
```

### 3. Automatic Application

When you run a backtest, the system:
1. Loads your strategy config
2. Detects the symbol (from config or CLI)
3. Looks up market profile for that symbol
4. Merges market settings into strategy config
5. Runs backtest with combined settings

## Usage Examples

### Test Same Strategy on Different Markets

**Crypto (BTCUSDT):**
```bash
python3 scripts/run_backtest.py \
  --strategy ema_crossover \
  --data data/raw/BTCUSDT-15m-2020.parquet
```

**Forex (EURUSD):**
```bash
python3 scripts/run_backtest.py \
  --strategy ema_crossover \
  --market EURUSD \
  --data data/raw/EURUSD-15m-2020.parquet
```

The system automatically:
- Applies EURUSD market profile (forex commissions, slippage)
- Uses forex-specific settings
- Strategy logic remains unchanged

### Override Market Settings

You can override any market setting via CLI:

```bash
python3 scripts/run_backtest.py \
  --strategy ema_crossover \
  --market EURUSD \
  --commission 0.0001 \
  --slippage 1.0 \
  --leverage 2.0
```

## Market Profile Structure

Market profiles define:

- **Trading Settings**:
  - `commission_rate`: Commission per trade (0.0004 = 0.04%)
  - `slippage_ticks`: Expected slippage
  - `leverage`: Leverage multiplier (1.0 = no leverage)

- **Precision Settings**:
  - `price_precision`: Decimal places for prices
  - `quantity_precision`: Decimal places for quantities

- **Market-Specific**:
  - `pip_value`: For forex (0.0001 for most pairs)
  - `contract_size`: For futures/forex
  - `min_trade_size`: Minimum trade size

## Asset Class Defaults

If a symbol isn't in market profiles, the system uses asset class defaults:

- **Crypto**: 0.04% commission, no slippage
- **Forex**: No commission, 0.5 pip slippage
- **Stock**: 0.1% commission, $0.01 slippage

## Best Practices

### 1. Keep Strategy Config Market-Agnostic

✅ **Good**: Strategy config contains only strategy logic
```yaml
moving_averages:
  ema5:
    enabled: true
    length: 5
```

❌ **Bad**: Hard-coding market-specific settings in strategy
```yaml
commissions: 0.0004  # Don't do this - use market profiles
```

### 2. Use Market Profiles for Market Settings

✅ **Good**: Define in `market_profiles.yml`
```yaml
markets:
  BTCUSDT:
    commission_rate: 0.0004
```

### 3. Test Across Multiple Markets

Always validate your strategy on:
- Different asset classes (crypto, forex, stocks)
- Different market conditions
- Different time periods

This helps identify:
- Strategy-specific vs market-specific performance
- Overfitting to one market
- Robustness across conditions

## Adding New Markets

To add a new market:

1. **Add to `market_profiles.yml`**:
```yaml
markets:
  YOURSYMBOL:
    exchange: your_exchange
    symbol: YOURSYMBOL
    asset_class: crypto  # or forex, stock
    commission_rate: 0.0004
    slippage_ticks: 0.0
    # ... other settings
```

2. **Use it**:
```bash
python3 scripts/run_backtest.py \
  --strategy ema_crossover \
  --market YOURSYMBOL \
  --data data/raw/YOURSYMBOL-data.parquet
```

## Benefits

1. **Reusability**: One strategy, multiple markets
2. **Consistency**: Same strategy logic across markets
3. **Realism**: Market-specific settings applied automatically
4. **Flexibility**: Easy to test "what if" scenarios
5. **Maintainability**: Market settings centralized

## Example: Testing Strategy on EURUSD

1. **Prepare data**: Put EURUSD data in `data/raw/EURUSD-15m-2020.parquet`

2. **Run backtest**:
```bash
python3 scripts/run_backtest.py \
  --strategy ema_crossover \
  --market EURUSD \
  --data data/raw/EURUSD-15m-2020.parquet
```

3. **System automatically**:
   - Loads MA alignment strategy
   - Applies EURUSD market profile (forex settings)
   - Uses forex commissions (0.0) and slippage (0.5 pips)
   - Runs backtest with combined settings

4. **Compare results**:
   - BTCUSDT: Crypto-specific performance
   - EURUSD: Forex-specific performance
   - Identifies if strategy is market-agnostic or crypto-specific

