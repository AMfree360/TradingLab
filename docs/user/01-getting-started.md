# Getting Started with Trading Lab

Welcome to Trading Lab! This guide will help you get started with the platform, even if you're new to programming.

## What is Trading Lab?

Trading Lab is a comprehensive platform for developing, testing, and validating trading strategies. It helps you:

- **Backtest** your trading ideas on historical data
- **Validate** strategies using professional-grade tests
- **Optimize** parameters to improve performance
- **Run live** strategies with proper risk management

## Installation

### Step 1: Install Python

Trading Lab requires Python 3.10 or higher. If you don't have Python installed:

1. Visit [python.org](https://www.python.org/downloads/)
2. Download Python 3.10 or newer
3. Run the installer
4. **Important**: Check "Add Python to PATH" during installation

### Step 2: Install Trading Lab

1. Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux)
2. Navigate to the Trading Lab folder:
   ```bash
   cd "path/to/Trading Lab"
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Verify Installation

Run this command to check if everything is installed correctly:

```bash
python scripts/run_backtest.py --help
```

You should see help text. If you get an error, check that Python is installed correctly.

## Your First Backtest

Let's run a simple backtest to make sure everything works:

```bash
python scripts/run_backtest.py --strategy ema_crossover --data data/raw/BTCUSDT_1m.csv
```

**Note**: You'll need to have data files in the `data/raw/` folder first. See the [Data Management Guide](DATA_MANAGEMENT.md) for how to get data.

## Next Steps

Once you've verified the installation works:

1. **Read the Complete Workflow Guide**: [02-complete-workflow.md](02-complete-workflow.md) - This provides a comprehensive step-by-step process for developing and validating strategies. **Highly recommended for all users.**

2. **Follow the Workflow**: The workflow guide covers:
   - Phase 1: Development (getting data, creating strategies, backtesting)
   - Phase 2: Validation (rigorous testing to ensure robustness)
   - Phase 3: Live Trading (deployment and monitoring)

3. **Explore Other Guides**: Use the [Documentation Index](README.md) to find specific guides as needed.

## Understanding the Project Structure

```
Trading Lab/
├── data/              # Your historical data goes here
│   ├── raw/          # Raw data files (CSV, Parquet)
│   ├── processed/    # Processed data (auto-generated)
│   └── manifests/    # Data metadata
├── strategies/        # Your trading strategies
│   └── ema_crossover/ # Example strategy
├── engine/           # Core backtesting engine
├── validation/       # Validation tests
├── metrics/          # Performance metrics
├── reports/          # Generated reports
├── scripts/          # Command-line tools
└── docs/             # Documentation (you are here!)
```

## Next Steps

Now that you're set up, here's what to do next:

1. **Learn about strategies**: Read [Strategy Development Guide](STRATEGY_DEVELOPMENT.md)
2. **Get data**: Read [Data Management Guide](DATA_MANAGEMENT.md)
3. **Run backtests**: Read [Backtesting Guide](BACKTESTING.md)
4. **Validate strategies**: Read [Validation Guide](VALIDATION.md)
5. **Understand results**: Read [Reports and Interpretation Guide](REPORTS.md)

## Common Issues

### "Python not found" error

**Solution**: Make sure Python is installed and added to PATH. Try using `python3` instead of `python` on Mac/Linux.

### "Module not found" error

**Solution**: Make sure you've installed dependencies:
```bash
pip install -r requirements.txt
```

### "Data file not found" error

**Solution**: Make sure your data file exists and the path is correct. Use absolute paths if relative paths don't work.

## Getting Help

If you encounter issues:

1. Check the relevant documentation guide
2. Review error messages carefully
3. Check that all dependencies are installed
4. Verify your data files are in the correct format

## What's Next?

Once you're comfortable with the basics, explore:

- Creating your own strategies
- Running validation tests
- Optimizing parameters
- Setting up live trading

Happy trading!

