# Trading Lab

[![CI](https://github.com/AMfree360/Dev/actions/workflows/ci.yml/badge.svg)](https://github.com/AMfree360/Dev/actions/workflows/ci.yml)

Trading Lab is a reproducible research and backtesting toolkit for developing, validating, and deploying trading strategies.

Key locations

- User docs: [docs/user/README.md](docs/user/README.md)
- Developer docs: [docs/dev/README.md](docs/dev/README.md)
- Example strategies: [strategies/](strategies/)

Quick start

```bash
# create and activate virtualenv
python3 -m venv .venv
. .venv/bin/activate

# install deps
pip install -r requirements.txt

# run tests
pip install pytest
pytest -q

# run a smoke backtest (example)
python3 scripts/run_backtest.py --strategy ema_crossover --data data/processed/BTCUSDT-15m-2023.parquet --market BTCUSDT --start-date 2023-01-01 --end-date 2023-03-31
```

CI

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` which lints with `ruff` and runs the test suite. An optional smoke backtest is available behind the `RUN_SMOKE` flag for manual dispatches.

Monte Carlo defaults

The validation suite defaults to `1000` Monte Carlo iterations per engine. This is the recommended minimum for stable p-value estimates; increase iterations for final validation runs as needed.

Contributing

See [docs/dev/README.md](docs/dev/README.md) for developer guidelines, coding style, and how to run the test suite locally.

License

This project is private/internal. Check with the repository owner for licensing and distribution policies.
# TradingLab
