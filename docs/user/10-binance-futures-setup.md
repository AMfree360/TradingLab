# Binance Futures Paper Trading Setup Guide

This guide explains how to set up and run paper trading with Binance Futures for BTCUSDT.

## Prerequisites

1. ✅ **API Credentials Ready**: You mentioned your API credentials are ready
2. ✅ **Strategy Validated**: Your strategy should pass backtesting and validation
3. ✅ **Python Environment**: Python 3.8+ with required packages

## Step 1: Install Dependencies

```bash
pip install python-binance
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Step 2: Set Up API Credentials

Create a `.env` file in the project root (copy from template below):

```bash
# Create .env file
cat > .env << EOF
# Binance API Credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Trading Settings (optional, can override via command line)
PAPER_MODE=true
INITIAL_CAPITAL=10000.0
MAX_DAILY_LOSS_PCT=5.0
MAX_DRAWDOWN_PCT=15.0
EOF
```

**Important**: 
- Replace `your_api_key_here` and `your_api_secret_here` with your actual credentials
- The `.env` file is already in `.gitignore` - never commit it to git!

## Step 3: Verify Binance API Permissions

### For Paper Trading (--paper-mode)
- **Read-only API key** (safest) - Only fetches price data, no trading
- No trading permissions needed (orders are simulated)

### For Testnet (--testnet) - **REAL ORDERS**
- **Enable Futures Trading** permission required
- API key must have futures trading access
- **NEVER enable** "Enable Withdrawals"
- Consider IP whitelisting for security
- Uses Binance testnet (test funds, not real money)

### For Live Trading (no flags) - **REAL MONEY**
- **Enable Futures Trading** permission required
- **NEVER enable** "Enable Withdrawals"
- IP whitelisting strongly recommended
- Uses Binance live exchange (real money at risk!)

## Step 4: Run Paper Trading

### Basic Paper Trading Command (Simulated)

```bash
python3 scripts/run_live.py \
  --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --paper-mode \
  --timeframe 15m \
  --capital 10000.0
```

**Note**: Paper mode simulates orders - no real API calls for trading (only for price data).

### Testnet Command (Real Orders on Testnet)

```bash
python3 scripts/run_live.py \
  --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --testnet \
  --timeframe 15m \
  --capital 10000.0
```

**Note**: Testnet places **REAL orders** on Binance testnet. Requires API credentials.

### Command Options

- `--strategy`: Strategy name (must match folder in `strategies/`)
- `--market`: Market symbol (use `BTCUSDT_FUTURES` for futures)
- `--paper-mode`: Enable paper trading (simulated, no real orders)
- `--timeframe`: Trading timeframe (15m, 1h, 4h, etc.)
- `--capital`: Starting capital (default: 10000.0)
- `--max-daily-loss-pct`: Stop trading if daily loss exceeds this % (default: 5.0)
- `--max-drawdown-pct`: Stop trading if drawdown exceeds this % (default: 15.0)
- `--update-interval`: How often to check for signals in seconds (default: 60)

### Example: Paper Trading with Custom Settings

```bash
python3 scripts/run_live.py \
  --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --paper-mode \
  --timeframe 15m \
  --capital 5000.0 \
  --max-daily-loss-pct 3.0 \
  --max-drawdown-pct 10.0 \
  --update-interval 30
```

## Step 5: Monitor Paper Trading

The script will:
- ✅ Log all activity to `data/logs/live_trading_YYYYMMDD_HHMMSS.log`
- ✅ Display real-time status in console
- ✅ Show balance, positions, and P&L
- ✅ Automatically stop if safety limits are exceeded

### What to Monitor

1. **Balance**: Should track paper trading balance
2. **Positions**: Open positions and their P&L
3. **Trades**: Number of trades executed vs skipped
4. **Daily P&L**: Profit/loss for the day
5. **Errors**: Any API or execution errors

## Step 6: Test with Binance Testnet (Recommended Before Live)

Before going live, test with Binance testnet. **Testnet places REAL orders** on Binance's test environment:

```bash
python3 scripts/run_live.py \
  --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --testnet \
  --timeframe 15m \
  --capital 10000.0
```

### Testnet vs Paper Mode

| Feature | Paper Mode | Testnet |
|---------|-----------|---------|
| **Orders** | Simulated (no API calls) | Real API orders |
| **Exchange** | None (local simulation) | Binance testnet |
| **Funds** | Virtual | Test funds from Binance |
| **API Required** | No (for price data only) | Yes (full API access) |
| **Best For** | Strategy logic testing | Real execution testing |

**Important**: 
- Testnet uses **real API calls** and places **real orders** on Binance testnet
- You need valid API credentials with futures trading permissions
- Testnet orders are executed on Binance's test environment (not real money)
- This is the best way to test real execution before going live

## Understanding Paper Trading Mode

### What Paper Trading Does

- ✅ **Simulates orders**: No real orders placed on exchange
- ✅ **Tracks positions**: Maintains virtual positions
- ✅ **Calculates P&L**: Based on current market prices
- ✅ **Respects limits**: Safety limits still apply
- ✅ **Logs everything**: Full logging for analysis

### What Paper Trading Doesn't Do

- ❌ **Real execution**: Orders are simulated, not executed
- ❌ **Real slippage**: Uses current price, no real slippage
- ❌ **Real fills**: All orders fill immediately at current price
- ❌ **Real margin**: Margin calculations are simulated

### Limitations

Paper trading is useful for:
- Testing strategy logic
- Verifying signal generation
- Testing safety limits
- Understanding strategy behavior

But remember:
- Results may differ from live trading
- Real slippage and fills can affect performance
- Market conditions may be different

## Safety Features

The live trading engine includes several safety features:

1. **Daily Loss Limit**: Stops trading if daily loss exceeds threshold
2. **Drawdown Limit**: Stops trading if drawdown exceeds threshold
3. **Balance Checks**: Verifies sufficient balance before trades
4. **Margin Checks**: Ensures sufficient margin before opening positions
5. **Position Limits**: Prevents opening new positions if one already exists

## Troubleshooting

### "API credentials required"

**Solution**: Make sure `.env` file exists with `BINANCE_API_KEY` and `BINANCE_API_SECRET`

### "Market profile not found"

**Solution**: Use `BTCUSDT_FUTURES` (not `BTCUSD_FUTURES`) - note the "USDT" not "USD"

### "Failed to get current price"

**Solution**: 
- Check internet connection
- Verify API key has read permissions
- Check if Binance API is accessible from your location

### "Insufficient balance"

**Solution**: 
- Increase `--capital` value
- Reduce position sizes in strategy config
- Check if margin requirements are too high

## Next Steps

After successful paper trading:

1. **Monitor for 1-2 weeks**: Run paper trading to verify everything works
2. **Compare to backtest**: Results should be similar to backtest expectations
3. **Review logs**: Check for any issues or unexpected behavior
4. **Test edge cases**: Test with different market conditions
5. **Consider testnet**: Test with Binance testnet before going live

## Going Live (Future)

When ready for live trading:

1. ✅ Paper trading successful for 1-2 weeks
2. ✅ Results match backtest expectations
3. ✅ No errors or issues in logs
4. ✅ Safety limits tested and working
5. ✅ Start with small capital
6. ✅ Monitor closely initially

**Live trading command** (use with extreme caution):

```bash
python3 scripts/run_live.py \
  --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --timeframe 15m \
  --capital 1000.0 \
  --testnet  # Use testnet first!
```

## Summary

You now have:
- ✅ Binance Futures API adapter
- ✅ Live trading engine
- ✅ Paper trading support
- ✅ Safety limits and monitoring
- ✅ Complete setup guide

Start with paper trading to verify everything works before considering live trading!

