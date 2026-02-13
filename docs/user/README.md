# Trading Lab - User Documentation

[![CI](https://github.com/AMfree360/Dev/actions/workflows/ci.yml/badge.svg)](https://github.com/AMfree360/Dev/actions/workflows/ci.yml)

Welcome to Trading Lab! This documentation is designed for **users** who want to use the system to develop, test, and trade strategies. If you're new to trading or programming, you're in the right place - we'll explain everything step by step.

## 📚 Documentation Index

### 🧭 Why TradingLab
- **[Why TradingLab Exists](00-why-tradinglab.md)** - Philosophy, who it’s for, and what it protects you from

### 🚀 Getting Started
- **[Getting Started Guide](01-getting-started.md)** - Start here! Installation and your first backtest
- **[Complete Workflow Guide](02-complete-workflow.md)** - Step-by-step process from idea to live trading ⭐ **RECOMMENDED**

### 📊 Using the System
- **[Data Management Guide](03-data-management.md)** - How to get, organize, and manage historical data
- **[Strategy Development Guide](13-strategy-development.md)** - How to create a new strategy (user-friendly)
- **[GUI Launcher (No-Code Workflow)](15-gui-launcher.md)** - Create/edit/compile/backtest from the local web UI
- **[Configuration Guide](04-configuration.md)** - How to configure your strategies (detailed explanation of every setting)
- **[Backtesting Guide](05-backtesting.md)** - How to run backtests and understand results
- **[Validation Guide](06-validation.md)** - How to validate strategies (ensuring they're not just lucky)
- **[Optimization Guide](07-optimization.md)** - How to optimize strategy parameters (without overfitting)
- **[Reports Guide](08-reports.md)** - How to read and interpret all reports and metrics
- **[Research Layer (No-Code Strategy Builder)](14-research-layer.md)** - Write strategies in YAML and compile them into TradingLab

### 💰 Live Trading
- **[Live Trading Guide](09-live-trading.md)** - How to prepare and run strategies with real money (safety first!)

### 🛠️ Reference
- **[Scripts Reference](10-scripts-reference.md)** - Complete reference for all command-line tools
- **[Golden Path: Split Policy + Config Profiles](11-split-policy-and-profiles.md)** - Standardized, reproducible workflow (recommended)

## 🎯 Quick Start Path

**New to Trading Lab?** Follow this path:

1. **Read**: [Getting Started Guide](01-getting-started.md) - Install and verify everything works
2. **Read**: [Complete Workflow Guide](02-complete-workflow.md) - Understand the full process
3. **Follow**: The workflow guide step-by-step to develop your first strategy
4. **Reference**: Other guides as needed during development

## 📖 What Each Guide Covers

### Getting Started
- Installing Python and dependencies
- Verifying installation
- Running your first backtest
- Understanding the project structure
- Common issues and solutions

### Complete Workflow
- **Phase 1: Development** - Getting data, creating strategies, initial backtesting
- **Phase 2: Validation** - Rigorous testing to ensure robustness
- **Phase 3: Live Trading** - Deployment and monitoring
- Decision points and gates
- Common patterns and troubleshooting

### Data Management
- Where to get historical data
- How to download and organize data
- Data format requirements
- Consolidating multiple files
- Data quality checks

### Configuration
- Complete explanation of every configuration option
- How each setting affects your strategy
- Trade Management architecture (master config + strategy overrides)
- Common configuration patterns
- Best practices

### Backtesting
- How to run backtests
- Understanding backtest results
- What makes a good backtest
- Common scenarios and examples

### Validation
- Why validation matters
- Phase 1: Training validation (proving edge exists)
- Phase 2: Out-of-sample validation (proving edge persists)
- Phase 3: Stationarity analysis (knowing when to retrain)
- Understanding validation results

### Optimization
- When to optimize (and when not to)
- How to optimize safely
- Avoiding overfitting
- Optimization strategies

### Reports
- Reading backtest reports
- Reading validation reports
- Understanding all metrics (including Edge Latency)
- Interpreting equity curves
- Common patterns and what they mean
- [Edge Latency Metric Guide](edge-latency-metric.md) - Statistical edge detection

### Live Trading
- Prerequisites for going live
- Safety settings and risk management
- Setting up exchange APIs
- Monitoring and maintenance
- When to stop trading

### Scripts Reference
- Complete list of all scripts
- How to use each script
- Common workflows
- Command-line options

## 💡 Tips for Success

1. **Start Simple**: Begin with basic strategies and add complexity gradually
2. **Follow the Workflow**: Don't skip steps - each phase has a purpose
3. **Be Patient**: Good strategies take time to develop and validate
4. **Accept Failure**: Most strategies will fail validation - that's normal and good!
5. **Read Carefully**: Take time to understand each guide
6. **Ask Questions**: If something is unclear, re-read the relevant section

## 🆘 Getting Help

If you're stuck:

1. **Check the relevant guide** - Most questions are answered in the guides
2. **Review error messages** - They often tell you exactly what's wrong
3. **Check the workflow guide** - It covers common issues
4. **Verify your setup** - Make sure installation is correct
5. **Check data quality** - Many issues come from bad data

## 📝 Assumptions

This documentation assumes:

- **You're new to trading systems** - We explain everything from scratch
- **You're new to programming** - We explain code concepts simply
- **You want to learn** - We provide detailed explanations, not just commands
- **You want to do it right** - We emphasize best practices and avoiding common mistakes

## 🎓 Learning Path

**Week 1: Basics**
- Read Getting Started
- Read Complete Workflow
- Run your first backtest

**Week 2: Development**
- Read Data Management
- Read Configuration
- Create your first strategy
- Read Backtesting

**Week 3: Validation**
- Read Validation Guide
- Run validation tests
- Understand results

**Week 4: Advanced**
- Read Optimization
- Read Reports
- Refine your strategy

**When Ready: Live Trading**
- Read Live Trading Guide thoroughly
- Set up paper trading
- Start small
- Monitor closely

## 🔗 Related Documentation

- **Developer Documentation**: See `/docs/dev/` for technical details, architecture, and extending the system
- **Main README**: See `/README.md` for project overview

## 📚 Additional Resources

- **Example Strategies**: Check `strategies/` folder for working examples
- **Configuration Examples**: See `config/` folder for configuration templates
- **Scripts**: See `scripts/` folder for all available tools

---

**Remember**: Trading involves risk. Always validate thoroughly, start small, and never risk more than you can afford to lose. The documentation emphasizes safety and best practices - follow it!

Happy trading! 🚀

## CI & Running Tests

We run a lightweight GitHub Actions CI that lints and runs the test suite. The workflow is located at `.github/workflows/ci.yml`.

- **What the workflow does**:
	- Lint with `ruff`.
	- Run `pytest` across supported Python versions.
	- Optional smoke backtest (controlled by `RUN_SMOKE` environment variable in the workflow).

- **How to run locally**:

```bash
# create and activate virtualenv
python3 -m venv .venv
. .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# lint (ruff)
pip install ruff
ruff check .

# run tests
pip install pytest
pytest -q

# run a quick smoke backtest (example)
python3 scripts/run_backtest.py --strategy ema_crossover --data data/processed/BTCUSDT-15m-2023.parquet --market BTCUSDT --start-date 2023-01-01 --end-date 2023-03-31
```

- **Notes**:
	- The workflow's smoke backtest step is skipped by default; set `RUN_SMOKE=true` in a manual `workflow_dispatch` if you want the CI job to attempt a smoke run (make sure sample data is available in the repo or via artifacts).
	- For CI stability, avoid committing large sample data; instead, run smoke backtests locally or via a dedicated artifact store.

- **Monte Carlo guidance**: The validation system defaults to **1000 Monte Carlo iterations** per engine (permutation, bootstrap, randomized-entry). For reliable p-values use at least 1000 iterations; increase to 2000–5000 for noisy strategies or final validation runs.
