# TradingLab GUI Launcher Spec (Thin Wrapper)

## Purpose
TradingLab is currently CLI-first and that is a feature: it’s reproducible, scriptable, and deterministic.

This document specifies a **GUI Launcher** that helps non-technical users run the *same* TradingLab workflows using large buttons, file pickers, and guided steps.

**The GUI Launcher is not a new product** and must not become a second implementation of TradingLab logic.

## Non‑Negotiable Principles
- **Single source of truth**: the existing Python scripts remain canonical for parse/validate/compile/backtest/validation.
- **GUI runs commands, nothing else**: GUI must not re-implement parsing, compilation, backtesting, or validation logic.
- **Always show the exact command**: user can copy/paste; command is stored in the run bundle.
- **Same outputs**: GUI produces the same files/folders as CLI (specs, strategy folders, reports).
- **Safe-by-default**: AI remains optional and pre-validation only; no hidden “auto decisions” (especially symbols).

## Target Users (Personas)
1) **Non-technical user**
- Can use a mouse, struggles with typing/terminal.
- Wants “choose inputs → run → open report”.

2) **Support / mentor**
- Helps diagnose failures.
- Needs logs, command lines, and reproducible run bundles.

3) **Power user (secondary)**
- Might still prefer CLI, but appreciates “open report / compare runs / browse artifacts”.

## Scope
### In scope (MVP)
- Big-button launcher for common workflows
- Guided clarification UI (when parser asks questions)
- Real-time log view + clear success/failure
- Run history + open report/output folder
- Run bundle artifacts (command + logs + metadata)

### Explicitly out of scope
- A second strategy editor that bypasses StrategySpec / CLI
- A parallel config schema
- Live trading controls (until we have a dedicated safety review)
- Complex parameter sweep UI (can be added later, but not MVP)

## Primary Workflows
Each workflow is a **wizard** with a single “Run” button. Every step uses file pickers / dropdowns.

### 1) Draft Strategy (from controlled English, offline)
**User inputs**
- Strategy name
- Input text file (controlled English)
- Default entry timeframe (optional)
- Parse mode: strict|lenient
- Output spec path (optional)
- Compile checkbox

**Command mapping**
- Script: `scripts/build_strategy_from_english.py`
- Example:
  - `./.venv/bin/python scripts/build_strategy_from_english.py --name NAME --in FILE --entry-tf 1h --parse-mode strict --compile`

**Outputs**
- `research_specs/<name>.yml` (unless overridden)
- `strategies/<name>/` if `--compile`

### 2) Draft Strategy (from messy notes + AI, optional)
**User inputs**
- Strategy name
- Notes file
- Provider (initially: ollama)
- Model
- Default entry timeframe (optional)
- Parse mode: strict|lenient
- Max repairs (default: 2)
- Auto-resolve clarifications (checkbox)
- Max clarify repairs (default: 1)
- Compile checkbox

**Command mapping**
- Script: `scripts/build_strategy_from_notes_ai.py`
- Example:
  - `./.venv/bin/python scripts/build_strategy_from_notes_ai.py --name NAME --in notes.txt --provider ollama --model llama3.1 --entry-tf 1h --parse-mode lenient --max-repairs 2 --auto-resolve-clarifications --max-clarify-repairs 1 --compile`

**Outputs**
- `research_specs/<name>.yml` (unless overridden)
- `strategies/<name>/` if `--compile`
- AI artifact bundle under `data/logs/ai/`

### 3) Compile Strategy (from spec)
**User inputs**
- Strategy spec file (`.yml`)

**Command mapping**
- Script: `scripts/build_strategy_from_spec.py`

**Outputs**
- `strategies/<name>/`

### 4) Run Backtest
**User inputs**
- Strategy folder / strategy name
- Dataset file (parquet/csv)
- Start/end date (optional)
- Report visuals checkbox (on by default)

**Command mapping**
- Script: `scripts/run_backtest.py`
- Output: report bundle under `reports/`

### 5) Run Validation (training / OOS / final)
**User inputs**
- Strategy name
- “Which validation” selector: training | oos | final
- Report visuals checkbox

**Command mapping**
- Scripts:
  - training: `scripts/run_training_validation.py`
  - oos: `scripts/run_oos_validation.py`
  - final: `scripts/run_final_oos_test.py`

**Outputs**
- report bundle(s) under `reports/`

### 6) Download Data
**User inputs**
- Provider (binance_vision, binance_api, yfinance, stooq, csv_url)
- Symbol
- Interval
- Start/end date
- Output path
- Optional resample-to timeframe

**Command mapping**
- Script: `scripts/download_data.py`

**Outputs**
- raw and/or processed data files under `data/`

## Clarification UX (Critical)
When `parse_english_strategy(...)` returns clarifications, the GUI must:
- Display them as one question per screen (or one page with a simple list)
- Use radio buttons when options exist
- Preselect default when provided
- On submit, re-run the same pipeline step (apply clarifications → validate)

Notes:
- Clarifications without defaults (e.g. `market.symbol`) must be user-provided.
- “Auto-resolve clarifications” is AI-only and **defaults-only**; GUI must clearly label it.

## UI Layout (Text Wireframes)
### Home
Large buttons (single column):
- Draft (English)
- Draft (Notes + AI)
- Compile
- Backtest
- Validate
- Download Data
- View Runs

Footer:
- “Open reports folder”
- “Open data folder”

### Wizard pattern
- Step title: “1/3 Choose inputs”
- Big file picker buttons (no path typing)
- Advanced section collapsed by default
- Primary action button: “Run”
- Secondary: “Cancel”

### Run screen
- Status: Running / Success / Failed
- Live log panel
- Buttons after completion:
  - Open report
  - Open output folder
  - Copy command
  - Copy diagnostic bundle

### Runs screen
- Table: timestamp, workflow type, strategy, status
- Actions: Open report, Open folder, Copy command

## Run Bundles (Reproducibility + Support)
Each GUI-triggered run creates a bundle directory (suggested):
- `data/logs/gui_runs/<timestamp>_<workflow>_<strategy>/`

Bundle contents (minimum):
- `command.txt` (exact command executed)
- `inputs.json` (wizard inputs, sanitized)
- `stdout.log`, `stderr.log`
- `meta.json`:
  - timestamps, exit code
  - TradingLab git SHA (if available)
  - python executable path
  - platform info
- Optional: references/paths to created reports/specs

## Execution Architecture (Thin Wrapper)
### Core components
1) **Command Builder**
- Deterministically maps wizard inputs → argv list
- Never concatenates shell strings (avoid quoting issues)

2) **Runner**
- Spawns subprocess with working directory = repo root
- Streams stdout/stderr incrementally to UI
- Captures exit status and duration

3) **Opener**
- Opens HTML report in default browser
- Opens folders via OS-specific calls

### Error handling
- Non-zero exit code → “Failed” state + show logs + offer bundle export
- Common errors get friendly tips, but must still show raw output

## Security & Safety
- Default bind is localhost only (if web UI)
- No secrets are stored in run bundles
- If AI providers are used, the GUI must clearly show:
  - provider + model
  - artifacts location
  - that AI output is validated deterministically

## Desktop vs Local Web UI (Decision Factors)
This spec supports either implementation. The decision should be based on:

### Installation & “double-click” experience
- Desktop app can be truly one-click.
- Local web UI needs a launcher (desktop shortcut) to start the server and open a browser.

### Cross-platform
- Local web UI is naturally cross-platform once started.
- Desktop apps require packaging per OS.

### Sandboxing / security
- Desktop: no server port, simpler threat model.
- Web UI: must ensure localhost-only + CSRF considerations.

### Maintenance cost
- Desktop shell adds build/release complexity.
- Web UI adds browser/front-end complexity.

**Recommendation for de-risking**: implement MVP as a local web UI with a tiny launcher script, then optionally wrap in a desktop shell later if demand warrants.

## MVP Milestones
1) **MVP (2–3 flows)**
- Draft from English
- Backtest
- View runs + open report

2) **MVP+**
- Draft from notes + AI
- Compile
- Download data

3) **Beta**
- Validation flows
- Better run comparison surfacing (links into existing comparison tools)

## Testing Strategy
- Unit tests for command builder (input → argv)
- Integration test (optional): run a tiny backtest end-to-end on a small fixture dataset
- No duplication of engine tests in GUI repo/module
