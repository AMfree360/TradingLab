# Live Trading Guide

This guide explains how to prepare and run your strategy live with real money. **Read this carefully before going live!**

## ⚠️ Important Warnings

**Before going live, understand**:
- Trading involves real financial risk
- Past performance doesn't guarantee future results
- Start with small capital
- Monitor closely initially
- Have a plan for stopping if things go wrong

## Prerequisites for Live Trading

Before going live, your strategy must:

1. ✅ **Pass backtesting**: Good results on historical data
2. ✅ **Pass validation**: Walk-forward and Monte Carlo tests pass
3. ✅ **Be optimized** (if needed): Parameters tuned and re-validated
4. ✅ **Have realistic costs**: Commissions and slippage included
5. ✅ **Show consistency**: Results consistent across time periods

**Do NOT go live if**:
- ❌ Validation fails
- ❌ Test PF < 1.5
- ❌ High drawdowns (> 20%)
- ❌ Phase 1 fails the universal Monte Carlo robustness gate
- ❌ Inconsistent results

## Preparation Steps

### Step 1: Paper Trading (Recommended)

Before risking real money, test with paper trading:

1. Set up paper trading account
2. Run strategy in paper mode for 1-2 weeks
3. Monitor all trades
4. Compare results to backtest expectations
5. Fix any issues before going live

### Step 2: Start Small

**Golden Rule**: Start with capital you can afford to lose!

**Recommended approach**:
- Start with 10-20% of intended capital
- Run for 1-3 months
- If successful, gradually increase
- Never risk more than you can lose

### Step 3: Set Safety Limits

Configure safety limits in your strategy:

```yaml
live:
  paper_mode: false
  check_balance_before_entry: true
  max_daily_loss_pct: 5.0
  enable_auto_disable_on_drawdown: true
  max_drawdown_pct: 15.0
```

**Safety settings**:
- `max_daily_loss_pct`: Stop trading if daily loss exceeds this
- `max_drawdown_pct`: Stop trading if drawdown exceeds this
- `check_balance_before_entry`: Verify sufficient funds before trades

### Step 4: Set Up Monitoring

**Monitor**:
- Daily P&L
- Trade frequency
- Win rate
- Drawdown
- Any errors or issues

**Set up alerts** for:
- Large losses
- Unusual activity
- System errors
- Drawdown thresholds

## Live Trading Setup

### Exchange API Setup

#### Binance Setup

1. **Create API Key**:
   - Log into Binance
   - Go to API Management
   - Create new API key
   - **Important**: Enable only "Enable Spot & Margin Trading"
   - **Never enable**: Withdrawals

2. **Set Permissions**:
   - Read only: For paper trading
   - Spot trading: For live trading
   - **Never enable**: Futures or withdrawals

3. **IP Whitelist** (Recommended):
   - Add your server's IP address
   - Increases security

4. **Store Credentials Securely**:
   ```bash
   # Create .env file (never commit to git!)
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   ```

### Environment Configuration

Create a `.env` file in the project root:

```env
# Exchange API Keys
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Trading Settings
PAPER_MODE=false
INITIAL_CAPITAL=1000.0
MAX_DAILY_LOSS_PCT=5.0
MAX_DRAWDOWN_PCT=15.0

# Monitoring
ALERT_EMAIL=your_email@example.com
LOG_LEVEL=INFO
```

**Important**: Add `.env` to `.gitignore` to never commit secrets!

## Running Live

### Basic Live Trading Command

```bash
python scripts/run_live.py \
  --strategy my_strategy \
  --exchange binance \
  --symbol BTCUSDT \
  --paper-mode false
```

### Paper Trading First

**Always test in paper mode first**:

```bash
python scripts/run_live.py \
  --strategy my_strategy \
  --exchange binance \
  --symbol BTCUSDT \
  --paper-mode true
```

Run for at least 1-2 weeks in paper mode to verify everything works.

## Monitoring Live Trading

### Daily Checks

**Every day, check**:
1. **P&L**: Compare to expectations
2. **Trades**: Number and quality
3. **Drawdown**: Is it within limits?
4. **Errors**: Any system issues?
5. **Market conditions**: Are they normal?

### Weekly Review

**Every week, review**:
1. **Performance vs backtest**: Are results similar?
2. **Win rate**: Is it as expected?
3. **Risk metrics**: Are they acceptable?
4. **Strategy behavior**: Is it working as designed?

### Monthly Analysis

**Every month, analyze**:
1. **Full performance report**: Generate and review
2. **Compare to validation**: Are results consistent?
3. **Parameter review**: Do any need adjustment?
4. **Market changes**: Has market structure changed?

## Common Live Trading Issues

### Issue 1: Results Don't Match Backtest

**Possible causes**:
- Slippage higher than expected
- Commissions different
- Market conditions changed
- Data quality issues in backtest

**Solutions**:
- Review actual costs
- Check market conditions
- Verify data quality
- Adjust expectations

### Issue 2: Too Many/Few Trades

**Possible causes**:
- Market conditions different
- Signal generation issues
- Position limits too restrictive

**Solutions**:
- Review signal generation
- Check market conditions
- Adjust position limits if needed

### Issue 3: Higher Drawdowns

**Possible causes**:
- Market volatility increased
- Risk per trade too high
- Stop losses not working

**Solutions**:
- Reduce risk per trade
- Check stop loss execution
- Review market conditions
- Consider pausing if too high

### Issue 4: System Errors

**Possible causes**:
- API issues
- Network problems
- Code bugs
- Exchange maintenance

**Solutions**:
- Check exchange status
- Review error logs
- Test API connectivity
- Have backup plan

## Risk Management

### Position Sizing

**Never risk more than**:
- 1-2% per trade (recommended)
- 5% total exposure at once
- 10% maximum drawdown

### Stop Losses

**Always use**:
- Stop losses on every trade
- Trailing stops when appropriate
- Maximum loss limits

### Drawdown Limits

**Set hard limits**:
- Daily loss limit: 5%
- Weekly loss limit: 10%
- Total drawdown limit: 15-20%

**Action when limits hit**:
- Stop trading immediately
- Review what went wrong
- Fix issues before resuming
- Consider strategy revision

## When to Stop Trading

**Stop immediately if**:
- ❌ Daily loss limit hit
- ❌ Drawdown limit exceeded
- ❌ Strategy clearly broken
- ❌ Market conditions changed drastically
- ❌ System errors persist

**Pause and review if**:
- ⚠️ Results consistently worse than backtest
- ⚠️ Win rate dropping significantly
- ⚠️ Unusual market conditions
- ⚠️ Multiple losing days in a row

## Performance Tracking

### Track These Metrics

1. **Daily P&L**: Track every day
2. **Win Rate**: Should match backtest
3. **Profit Factor**: Should be > 1.5
4. **Max Drawdown**: Should stay within limits
5. **Trade Count**: Compare to expectations

### Compare to Backtest

**Regularly compare**:
- Actual vs expected P&L
- Actual vs expected win rate
- Actual vs expected trade frequency
- Actual vs expected drawdown

**If consistently different**:
- Review why
- Adjust expectations
- Consider strategy revision

## Optimization While Live

### When to Re-optimize

**Consider re-optimizing if**:
- Market conditions changed significantly
- Performance degraded consistently
- New data available (6+ months)
- Stationarity analysis suggests retraining

### How to Re-optimize

1. **Collect live data**: At least 3-6 months
2. **Add to historical data**: Combine with backtest data
3. **Re-run optimization**: Use updated dataset
4. **Re-validate**: Always validate optimized parameters
5. **Test in paper mode**: Before applying to live
6. **Gradual rollout**: Apply changes gradually

**Never**:
- ❌ Optimize on live data only
- ❌ Skip validation
- ❌ Apply changes without testing
- ❌ Optimize too frequently

## Best Practices

### 1. Start Small

- Begin with minimal capital
- Prove strategy works
- Gradually increase

### 2. Monitor Closely

- Check daily initially
- Set up alerts
- Review regularly

### 3. Keep Records

- Log all trades
- Track performance
- Document issues

### 4. Have a Plan

- Know when to stop
- Have exit criteria
- Plan for problems

### 5. Stay Disciplined

- Follow the strategy
- Don't override signals
- Trust the process

### 6. Continuous Improvement

- Review regularly
- Learn from mistakes
- Improve over time

## Troubleshooting

### "API Error" messages

**Solutions**:
- Check API key permissions
- Verify IP whitelist
- Check exchange status
- Review rate limits

### "Insufficient balance" errors

**Solutions**:
- Check account balance
- Reduce position size
- Verify risk settings

### "Strategy not generating signals"

**Solutions**:
- Check market conditions
- Verify data feed
- Review strategy logic
- Check configuration

## Next Steps

After going live:

1. **Monitor closely**: First few weeks are critical
2. **Track performance**: Compare to expectations
3. **Review regularly**: Weekly/monthly reviews
4. **Improve continuously**: Learn and adapt

## Summary

Live trading is the final step, but requires:
- ✅ Thorough preparation
- ✅ Small start
- ✅ Close monitoring
- ✅ Risk management
- ✅ Discipline

Remember: The goal is consistent, sustainable profits, not quick wins. Take it slow, be careful, and always prioritize capital preservation!

Good luck, and trade safely!

