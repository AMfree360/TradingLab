# Binance Testnet Trading Guide

## Quick Start: Real Orders on Testnet

To use **real orders on Binance testnet** (not paper trading):

```bash
python3 scripts/run_live.py \
   --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --testnet \
  --timeframe 15m \
  --capital 10000.0
```

## What is Testnet?

- ✅ **Real API calls** - Uses actual Binance API
- ✅ **Real orders** - Orders are placed on Binance testnet
- ✅ **Test funds** - Uses Binance's test environment (not real money)
- ✅ **Real execution** - Tests actual order execution, fills, slippage
- ✅ **Best practice** - Test real execution before going live

## Requirements

1. **API Credentials**: Set in `.env` file:
   ```bash
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   ```

2. **API Permissions**: Your API key must have:
   - ✅ "Enable Futures" permission
   - ❌ "Enable Withdrawals" (NEVER enable this)

3. **Testnet Account**: 
   - Visit https://testnet.binancefuture.com/
   - Create a testnet account (separate from main Binance account)
   - Get testnet API credentials from testnet account

## Important Notes

### Testnet vs Paper Mode

| | Paper Mode | Testnet |
|---|---|---|
| **Command** | `--paper-mode` | `--testnet` |
| **Orders** | Simulated | Real API orders |
| **Exchange** | None | Binance testnet |
| **API Calls** | Price data only | Full trading API |
| **Execution** | Instant simulated | Real order execution |
| **Slippage** | None | Real slippage |
| **Best For** | Strategy logic | Real execution testing |

### Cannot Use Both

You **cannot** use both `--paper-mode` and `--testnet` together. They are mutually exclusive:
- `--paper-mode`: Simulated trading (no real orders)
- `--testnet`: Real orders on testnet
- Neither flag: Real orders on live exchange (real money!)

## Getting Testnet API Credentials

1. Go to https://testnet.binancefuture.com/
2. Log in (or create testnet account)
3. Go to API Management
4. Create API key with "Enable Futures" permission
5. Copy API key and secret to your `.env` file

**Note**: Testnet API credentials are **different** from your main Binance account credentials.

## Example: Full Testnet Command

```bash
python3 scripts/run_live.py \
   --strategy ema_crossover \
   --market BTCUSDT_FUTURES \
   --testnet \
   --timeframe 15m \
   --capital 10000.0 \
   --max-daily-loss-pct 5.0 \
   --max-drawdown-pct 15.0 \
   --update-interval 60
```

## What to Expect

When running with `--testnet`:

1. **Real API Connection**: Connects to Binance testnet API
2. **Real Orders**: Orders are placed on testnet exchange
3. **Real Execution**: Orders execute on testnet (may have real slippage)
4. **Test Funds**: Uses testnet account balance (not real money)
5. **Full Logging**: All orders and executions are logged

## Monitoring Testnet Trading

- Check testnet account: https://testnet.binancefuture.com/
- View positions and orders in testnet dashboard
- Compare with logs in `data/logs/live_trading_*.log`

## Troubleshooting

### "API credentials required"
- Make sure `.env` file has `BINANCE_API_KEY` and `BINANCE_API_SECRET`
- Use **testnet** API credentials (not main account)

### "Failed to place order"
- Check API key has "Enable Futures" permission
- Verify you're using testnet API credentials
- Check testnet account has sufficient test funds

### "Cannot use both --paper-mode and --testnet"
- Remove one of the flags
- Use `--testnet` for real orders on testnet
- Use `--paper-mode` for simulated trading

## Next Steps

After successful testnet trading:

1. ✅ Monitor for 1-2 weeks on testnet
2. ✅ Verify orders execute correctly
3. ✅ Check slippage and fills match expectations
4. ✅ Compare results to backtest
5. ✅ Only then consider live trading (remove `--testnet` flag)

## Safety Reminder

- ✅ Testnet = Real orders but test funds (safe)
- ⚠️ Live = Real orders with real money (risky!)
- Always test on testnet first before going live

