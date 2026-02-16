# GUI Launcher (No-Code Workflow)

TradingLab includes a thin local web UI (“Launcher”) that runs the same scripts as the CLI.

The goal is: **create strategies, iterate on specs, and run backtests without touching code**.

## Install + Run

```bash
./.venv/bin/python -m pip install -r requirements-gui.txt
./.venv/bin/python scripts/run_gui_launcher.py --host 127.0.0.1 --port 8000
```

Then open:
- `http://127.0.0.1:8000/`

## Key Pages

- `/` Home
- `/strategies` Strategy hub
  - lists strategy specs from `research_specs/` (editable)
  - lists code strategies from `strategies/` (reference)
- `/strategies/spec/<name>` Spec editor
  - edit YAML
  - validates using `StrategySpec`
  - “Save + compile” builds the strategy into `strategies/<name>/`
- `/backtest` Backtest form
  - supports file upload dataset + advanced options
- `/runs` Run bundles
  - inputs/command/stdout/stderr + report links
  - run detail includes “Rerun (same settings)” when available

## Golden Path Workflow

1. Create a strategy (Guided Builder v2 or English)
2. Open `/strategies` → pick your spec → edit YAML
3. Click **Save + compile**
4. Run a backtest from `/backtest`
5. Open `/runs/<run_name>` to inspect logs + report
6. Iterate with **Rerun (same settings)** and spec edits

Notes:
- Guided strategy creation uses Builder v2; editing is either Builder v2 (when available) or YAML.

## Where Files Are Stored

- Strategy specs (editable source of truth): `research_specs/<name>.yml`
- Compiled strategies: `strategies/<name>/`
- GUI run bundles: `data/logs/gui_runs/<timestamp>_<workflow>_<strategy>/`
  - `inputs.json` (what you asked for)
  - `command.txt` (what the GUI executed)
  - `stdout.log` / `stderr.log`
  - `spec.yml` (when relevant)

## Notes

- The GUI is designed to be **non-interactive**: it passes `--auto-resample` on backtests so runs cannot hang waiting for terminal input.
- If spec edits do not change your backtest results, you likely forgot to **Save + compile** before rerunning.
