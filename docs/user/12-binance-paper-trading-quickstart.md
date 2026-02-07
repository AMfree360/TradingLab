# Binance Futures Paper Trading - Quick Start

## ✅ Files Created

1. **`adapters/execution/binance_futures.py`** - Binance Futures API adapter with paper trading support
2. **`engine/live_trading_engine.py`** - Live trading engine that executes strategies
3. **`scripts/run_live.py`** - Main script to run live/paper trading
4. **`docs/user/10-binance-futures-setup.md`** - Complete setup guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install python-binance
```

### 2. Set Up API Credentials

Create `.env` file in project root:

```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

### 3. Run Paper Trading

```bash
python3 scripts/run_live.py \
  --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --paper-mode \
  --timeframe 15m \
  --capital 10000.0
```

## 📋 Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--strategy` | Strategy name (required) | - |
| `--market` | Market symbol (required) | - |
| `--paper-mode` | Enable paper trading | False |
| `--testnet` | Use Binance testnet | False |
| `--timeframe` | Trading timeframe | 15m |
| `--capital` | Starting capital | 10000.0 |
| `--max-daily-loss-pct` | Max daily loss % | 5.0 |
| `--max-drawdown-pct` | Max drawdown % | 15.0 |
| `--update-interval` | Update interval (seconds) | 60 |

## ⚠️ Important Notes

1. **Use `BTCUSDT_FUTURES`** (not `BTCUSD_FUTURES`) - note the "USDT"
2. **Paper mode** simulates orders - no real money at risk
3. **Testnet** uses real API but with test funds
4. **Live trading** requires explicit confirmation and real API credentials

## 📚 Full Documentation

See `docs/user/10-binance-futures-setup.md` for complete setup instructions.

## 🔒 Safety Features

- Daily loss limit (default: 5%)
- Drawdown limit (default: 15%)
- Balance checks before trades
- Margin validation
- Position limits

## 📊 Monitoring

- Logs saved to `data/logs/live_trading_YYYYMMDD_HHMMSS.log`
- Real-time status in console
- Balance, positions, and P&L tracking

## 🎯 Next Steps

1. Run paper trading for 1-2 weeks
2. Monitor results and compare to backtest
3. Test with Binance testnet (optional)
4. Consider live trading only after thorough testing

