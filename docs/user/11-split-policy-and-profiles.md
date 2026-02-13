# Golden Path: Split Policy + Config Profiles

This guide explains the *standardized*, reproducible workflow in TradingLab:

- A single **split policy** defines your training / walk-forward OOS / final holdout date ranges.
- A **config profile** freezes strategy parameters so Phase 1, Phase 2, and final OOS all run the *same config*.

If you follow this “golden path”, you get:
- Time-series safe evaluation (no accidental leakage)
- Comparable results across runs
- Reproducible validation (manifests + phase locks)

## Quick glossary

- **Split policy**: The single source of truth for date ranges and walk-forward settings, defined in `config/data_splits.yml`.
- **Profile**: A named YAML override file that freezes strategy parameters at `strategies/<strategy>/configs/profiles/<profile>.yml`.
- **Phase 1**: Training validation gate (iterative). Tune and iterate here.
- **Phase 2 (WF OOS)**: Walk-forward out-of-sample gate. Treat as *consumed* (don’t iterate based on it).
- **Holdout / Final OOS**: The last untouched slice (e.g., 2025). Run once after Phase 2 passes.
- **Manifest / Phase lock**: A recorded dataset identity used to prevent rerunning a phase on a different underlying dataset slice without an explicit reset.

## At a glance (diagram)

```
Processed dataset (e.g., BTCUSDT 4h)
  |
  v
Split policy (config/data_splits.yml)
  - backtest/dev slice (iterative)
  - phase1 slice (iterative gate)
  - phase2 slice (OOS walk-forward)
  - final holdout slice (one-shot)
  |
  v
Optional tuning (Phase 1 slice only)  ---> writes profile: strategies/<s>/configs/profiles/<name>.yml
  |
  v
Phase 1 validation (training gate)    ---> must PASS before touching Phase 2 “for real”
  |
  v
Phase 2 validation (WF OOS gate)      ---> treat as consumed/one-shot
  |
  v
Final OOS holdout (2025, one-shot)    ---> run once after Phase 2 passes
```

---

## 1) Split policy (required for standardized runs)

**File**: `config/data_splits.yml`

The split policy is the source of truth for:
- Backtest/dev slice (iterative)
- Phase 1 slice (training validation gate)
- Phase 2 slice + walk-forward windowing (true OOS gate)
- Final OOS slice (locked holdout)

### How it’s enforced

By default, the main scripts enforce the policy:
- `scripts/run_backtest.py`
- `scripts/run_training_validation.py`
- `scripts/run_oos_validation.py`
- `scripts/run_final_oos_test.py`

If you pass ad-hoc `--data`, `--start-date`, or `--end-date` that don’t match the policy, the script will error unless you explicitly opt out with:
- `--override-split-policy`

Use `--override-split-policy` for diagnostics only.

---

## 2) Config profiles (freeze the strategy config)

Config profiles let you save “the chosen parameters” as a named YAML file.

**Location**:
- `strategies/<strategy>/configs/profiles/<profile>.yml`

**How profiles are applied**:
- Base config loads from: `strategies/<strategy>/config.yml`
- Market config may apply (if present)
- Profile overrides are deep-merged on top of the base config

### Why profiles matter

Without a profile, it’s easy to accidentally change parameters between Phase 1 and Phase 2.

With a profile, you can:
- Tune on Phase 1 slice (iterative)
- Lock the chosen config
- Run Phase 1 → Phase 2 → final holdout using the same named profile

---

## 3) The “Golden Path” workflow

### Step A — Prepare data

1) Download raw data (example):

```bash
./.venv/bin/python scripts/download_data.py \
  --provider binance_vision \
  --market-type spot \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2019-01-01 \
  --end 2024-12-31 \
  --resample-to 4h \
  --processed-output data/processed/BTCUSDT-4h-2019-2024-v2.parquet
```

2) Ensure `config/data_splits.yml` points at the processed dataset and the policy ranges you intend to use.

### Step B — Dev backtest (iterative)

```bash
./.venv/bin/python scripts/run_backtest.py \
  --strategy <strategy_name>
```

This uses the policy’s backtest/dev range by default.

### Step C — Optional: tune on Phase 1 slice (iterative)

If you need better parameters (or want to explore), tune on Phase 1 only and write a profile.

Example (strategy-specific tuner):

```bash
./.venv/bin/python scripts/tune_donchian_breakout.py \
  --max-evals 300 \
  --top-n 10 \
  --write-profile phase1_tuned
```

### Step D — Phase 1 (training validation gate)

```bash
./.venv/bin/python scripts/run_training_validation.py \
  --strategy <strategy_name> \
  --config-profile phase1_tuned \
  --mc-iterations 1000
```

Phase 1 is the robustness gate. Iterate (including tuning) until it passes.

### Step E — Phase 2 (walk-forward OOS gate)

```bash
./.venv/bin/python scripts/run_oos_validation.py \
  --strategy <strategy_name> \
  --config-profile phase1_tuned
```

Important:
- Phase 2 should be treated as one-shot/consumed.
- If you’re still iterating on logic/params, go back to Phase 1.

### Step F — Final OOS holdout (run once)

```bash
./.venv/bin/python scripts/run_final_oos_test.py \
  --strategy <strategy_name> \
  --config-profile phase1_tuned
```

---

## 4) Manifests + phase locks (why reruns can be blocked)

TradingLab writes dataset manifests under `data/manifests/` and locks each phase to the exact dataset slice identity.

This prevents invalid “compare apples to oranges” validation:
- If the split policy changes (dates, dataset file, etc.), Phase 1/2/final may refuse to proceed until you explicitly reset validation state.

Use resets only for development/testing and treat OOS/holdout as consumed once you’re running the real pipeline.

---

## 5) Common mistakes (and how to avoid them)

### Mistake: Changing splits mid-validation

If you edit `config/data_splits.yml` (dates, dataset file, policy name) after you’ve already run Phase 1/2, the next run may fail with a message like “Dataset manifest changed”.

That’s expected: phase locks exist to prevent invalid comparisons.

**Fix (dev/testing only):** reset the relevant phase state using:

```bash
./.venv/bin/python scripts/reset_validation_state.py --strategy <strategy_name> --data <dataset_file> --phase 1 --yes
./.venv/bin/python scripts/reset_validation_state.py --strategy <strategy_name> --data <dataset_file> --phase 2 --yes
```

### Mistake: Using `--override-split-policy` as the default

`--override-split-policy` is for diagnostics. If you use it routinely, you can accidentally validate on the wrong period or leak OOS/holdout data into iteration.

**Rule of thumb:**
- Use split policy enforcement for “real” results.
- Use override only for one-off debugging (e.g., “how many trades did we get in 2022?”).

### Mistake: Retuning after Phase 2 (data mining)

Once you run Phase 2 “for real”, don’t go back and retune based on those results and rerun Phase 2. That turns OOS into training.

**Correct loop:**
Iterate on Phase 1 (including tuning) until strong → then run Phase 2 once → then run final holdout once.

### Mistake: Not freezing the config

If you don’t use a profile, it’s easy to make small config changes between phases without realizing it.

**Fix:** always run Phase 1/2/final with `--config-profile <name>` once you’ve chosen a candidate.

### Mistake: Profile contains only `params`

Profiles can override *any* config subtree (risk, filters, take profit, trailing, partial exits, etc.). If your strategy relies on behavior outside `params`, ensure the profile captures it.

**Tip:** treat the profile as “the frozen strategy config” for validation.

---

## FAQ

### Why is Phase 2 “consumed” / treated as one-shot?

Because Phase 2 is your *out-of-sample* gate. If you repeatedly rerun Phase 2 and tweak the strategy based on those results, you effectively turn OOS into training data (data mining).

In practice:
- Iterate freely on dev backtests and Phase 1.
- Run Phase 2 when you’re ready to accept the result.
- Only after Phase 2 passes, run the final holdout once.

### What is a dataset manifest?

A dataset manifest is a deterministic record of “what exact slice of data was used” (dataset identity + date range + other invariants). It’s written under `data/manifests/` and printed during runs.

Manifests enable:
- Reproducibility (same inputs → comparable results)
- Phase locks (Phase 1/2/holdout refuse to proceed if the dataset identity changes)

### Why did I get “Dataset manifest changed… refusing to proceed”?

This means the phase lock detected that the dataset slice identity is different from the prior run for that phase (common causes: changed split policy dates, changed dataset file, regenerated processed data).

**What to do:**
- If you’re in development/testing: reset the phase state (see reset commands above) and rerun.
- If you’re trying to preserve true OOS/holdout semantics: do not reset and rerun; treat it as a new experiment with a new policy/dataset.

### Should I always use `run_validation.py`?

Not required. For the standardized workflow, the phase-specific scripts are preferred:
- Phase 1: `scripts/run_training_validation.py`
- Phase 2: `scripts/run_oos_validation.py`
- Holdout: `scripts/run_final_oos_test.py`

`scripts/run_validation.py` is useful when you explicitly want a combined pipeline runner, but it still follows the same split-policy enforcement rules.
