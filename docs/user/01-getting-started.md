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

## Recommended for New Users: Guided Strategy Builder (GUI)

If you're new to systematic trading (or you want to explore ideas quickly), the fastest way to start is the **GUI Launcher**.

1. Start the GUI:
   ```bash
   python scripts/run_gui_launcher.py --reload
   ```
2. Open the Launcher in your browser (the terminal will show the URL).
3. Click **Create Strategy (Guided)**.

Builder v2 (**Context + Trigger**) is the guided strategy creation experience.

The guided builder walks you through the core components of a backtestable strategy:
- Market / symbol
- Holding style + session controls
- Entry conditions
- Stops + risk sizing
- Advanced (optional): trade management + calendar filters + execution options

Power users can skip the wizard and edit specs directly in YAML (GUI: **Edit (advanced)**).

After you review the YAML preview, you can run a build/backtest from the same UI.

### Builder v2 visual previews (highly recommended)

In Builder v2 Step 4, Trading Lab includes two chart previews to help you validate your logic before you click **Review**:

- **Context visual (preview)**: entry-timeframe candles with the selected primary context overlaid (e.g., MA stack / Bollinger channel / ATR% line) and regime shading.
- **Setup visual (preview)**: entry-timeframe candles with context shading and markers where **ALL** of your **Context + Signals + Triggers** align.

Notes:

- These previews use the newest dataset from your GUI runs under `data/logs/gui_runs/**/dataset_*`.
   - If the charts say “No dataset found…”, run any backtest from the GUI (or upload a dataset) once, then return to the builder.
- If **Context TF** differs from **Entry TF**, context is evaluated on the context timeframe and forward-filled to the entry timeframe for plotting and evaluation alignment (so shading can look “stepped”).
- The setup chart may include optional marker layers (signals-only / triggers-only) that are hidden by default; use the Plotly legend to toggle them on.

### Builder v2 Step 5: Execution option — exit if context invalidates

In Step 5 (Advanced), you can optionally enable **Exit if context invalidates**.

This is useful for regime-based strategies where your **context filter** defines when trading is allowed (e.g., trend is bullish). If the context flips from valid → invalid while you’re in a trade, the strategy can react automatically:

- **Immediate exit**: emits an exit intent when context flips false, closing the position.
- **Tighten stop**: emits a “tighten stop” intent (no instant close). The position remains open, but the stop can be tightened to reduce risk.

Notes:

- This only matters if you’re using a context filter (Step 4) and/or a higher **Context TF**. If you have no context, there’s nothing to invalidate.
- Default is **disabled** (recommended until you understand the trade-offs).

If you’re editing specs directly, the YAML looks like:

```yaml
execution:
   exit_if_context_invalid:
      enabled: true
      mode: immediate   # or: tighten_stop
```

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

1. **Learn about strategies**: Read [Strategy Development Guide](13-strategy-development.md)
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

