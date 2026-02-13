# Why TradingLab exists

TradingLab is built for a specific kind of trader:

- **Self-taught** (not institutional, not plugged into a quant desk playbook)
- Often **discretionary-first**, but drawn toward systems because only systems can be evaluated honestly
- Wants trading decisions that **make mathematical sense** and can be tested, not “felt”

If that’s you, the biggest danger usually isn’t *lack of effort*.

It’s spending months or years on something that looks good **right now**, without ever proving:

- whether the edge was real,
- whether it survives different regimes,
- whether costs and execution assumptions destroy it,
- or whether you’ve accidentally “cheated” (lookahead, leakage, or post-hoc storytelling).

TradingLab is an attempt to encode the hard-earned lessons of arriving at systematic thinking the long way.

## The core problem: false confidence

Many traders can spend a long time trading a “strategy” without ever seeing the actual statistics of it.

A simple failure mode:

- You start trading a discretionary idea in a year where the market regime happens to align with it.
- Prior years would have failed, but you never tested them properly.
- The idea looks like alpha, so you scale it.
- Eventually, the regime shifts and the same idea becomes a life-changing drawdown.

TradingLab exists to reduce the odds of that happening by giving you a **compass**:

- a workflow that makes it hard to fool yourself,
- a validation gate that forces you to confront uncertainty,
- and artifacts (reports + manifests + configs) that you can review later.

This is not about guaranteeing profitability. It’s about reducing self-deception.

## What TradingLab is trying to enforce

### 1) Reproducibility by default

TradingLab encourages a policy-driven workflow where you can’t quietly change the data slice and accidentally “improve” results.

Key ideas in the repo:

- **Split policies** (standard train/OOS/holdout ranges)
- **Dataset locks/manifests** so a phase fails if the dataset identity changes
- “One-shot” semantics for **OOS / holdout** so you treat those results like scarce, high-signal evidence

### 2) Validation beyond a single backtest

A single backtest can be lucky.

TradingLab’s pipeline is built to push you toward:

- robustness checks (Monte Carlo suite)
- sanity checks around suitability
- consistent reporting so failures are legible and actionable

### 3) Clear assumptions about timing

TradingLab uses an opinionated default execution model:

- **decide at bar close, fill at next bar open**

This is a deliberate “anti-lookahead” choice. It makes many strategies look worse in the short term, but more believable in the long term.

### 4) Visual confirmation when it matters

Numbers can hide bugs.

TradingLab reports can include a **Trade Chart** that shows:

- price (candles)
- selected indicator overlays
- entries, exits, stop-loss segments, partial exits

The goal isn’t aesthetics — it’s catching “that’s not what I meant” errors early.

## AI (optional): where it fits safely

TradingLab does **not** require AI to work. In fact, the core workflow intentionally prefers:

- deterministic parsing and compilation
- schema validation
- reproducible artifacts

That said, in 2026 AI can be a high-leverage *assistant* if it’s used **before** validation and **outside** execution.

The safe shape is:

1) You describe an idea in messy notes
2) AI proposes a clean, controlled-English version (or a draft YAML spec)
3) TradingLab validates it (schema + sanity rules) and compiles it deterministically

Good AI use-cases that complement the existing English → YAML parser:

- **Drafting**: turn unstructured notes into controlled-English that the parser already understands
- **Clarifying questions**: suggest what the system should ask next (the final questions/answers still get applied deterministically)
- **Report explanation / anomaly triage**: summarize what changed between runs and point to likely failure modes
- **Guardrail warnings**: flag suspicious assumptions (tiny sample size, too-perfect fills, misaligned timeframes, unrealistic costs)

Bad AI use-cases (for this repo’s mission):

- AI directly deciding entries/exits in live trading (hard to audit, hard to reproduce)
- “find me a profitable strategy” black-box searches (usually just overfitting)

If/when AI is integrated, the non-negotiables should be:

- AI output must pass schema validation (no “close enough”)
- prompts + model/version + output are stored as artifacts for reproducibility
- AI never sits on the critical execution path (it can draft specs; the engine still runs deterministically)

## Who should use TradingLab

TradingLab is a good fit if you want:

- a disciplined workflow for idea → backtest → validation → (eventual) live
- guardrails against accidental leakage and overconfidence
- a local, inspectable codebase you can evolve as you learn

## Who shouldn’t (yet)

You’ll likely prefer other platforms if you need:

- tick/order-book simulation, or high-frequency execution realism
- a huge indicator ecosystem out of the box
- a hosted all-in-one platform with brokerage, data, and deployment handled for you

## What “success” looks like here

Success is:

- rejecting most ideas quickly and correctly,
- developing a smaller number of strategies that survive harsh validation,
- and building confidence that you’re not mistaking luck for skill.

If you’re a lone wolf doing this without institutional scaffolding, that’s a meaningful edge.
