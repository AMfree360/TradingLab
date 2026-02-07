**Environment variables & secrets**

- Copy `.env.example` to `.env` and fill with your real credentials.
- NEVER commit your `.env` or any file containing secrets to Git. `.env` is listed in `.gitignore`.

Recommended variables (placeholders in `.env.example`):
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET` (optional)
- `DEFAULT_TIMEZONE`, `LOG_LEVEL`

CI / GitHub Actions:
- Store credentials in repository or organization Secrets, then reference them in workflows.
- Example (GitHub Actions):

  - name: Set env
    run: |
      echo "BINANCE_API_KEY=${{ secrets.BINANCE_API_KEY }}" >> $GITHUB_ENV
      echo "BINANCE_API_SECRET=${{ secrets.BINANCE_API_SECRET }}" >> $GITHUB_ENV

If secrets were accidentally committed:
1. Revoke and rotate the exposed keys immediately.
2. Remove from git history (BFG or `git filter-repo`) and force-push cleaned history.
3. Inform collaborators and rotate any dependent credentials.

Prevention:
- Keep `.env` in `.gitignore`.
- Commit a `.env.example` with placeholders so contributors know required keys.
- Install a pre-commit hook (`detect-secrets` or `gitleaks`) to block accidental commits.

Pre-commit hooks

- We include a `.pre-commit-config.yaml` that runs basic hygiene checks and `detect-secrets`.
- To enable locally, install `pre-commit` and run:

```bash
python3 -m pip install --user pre-commit
pre-commit install
```

- After installation, `detect-secrets` will scan staged files and block commits containing secrets.
  You can run the full scan manually:

```bash
pre-commit run --all-files
```
