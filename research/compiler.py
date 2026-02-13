from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from config.schema import validate_strategy_config
from research.dsl import EvalContext, compile_condition, extract_indicator_requests, extract_timeframe_refs
from research.indicators import ensure_indicators
from research.spec import StrategySpec


_STRATEGY_TEMPLATE = """\
\"\"\"AUTO-GENERATED strategy from TradingLab research layer.

This file is generated from a StrategySpec YAML and is intended to be treated
as build output. Edit the spec and re-generate instead of hand-editing.
\"\"\"

from __future__ import annotations

from typing import Dict, Optional

import json
import pandas as pd

from strategies.base.strategy_base import StrategyBase

from research.dsl import EvalContext, compile_condition, extract_indicator_requests
from research.indicators import ensure_indicators


class GeneratedResearchStrategy(StrategyBase):
    def __init__(self, config):
        super().__init__(config)

        self._spec = json.loads(__SPEC_JSON__)

        self._context_tf = self._spec.get("context_tf") or self._spec.get("entry_tf")

        # Compile condition callables per side/timeframe
        self._compiled = {"long": [], "short": []}
        self._compiled_context = {"long": [], "short": []}
        self._compiled_exits = {"long": [], "short": []}
        for side in ["long", "short"]:
            side_spec = self._spec.get(side)
            if not side_spec or not side_spec.get("enabled", True):
                continue
            for cond in side_spec.get("conditions_all", []):
                item = {"tf": cond["tf"], "expr": cond["expr"], "fn": compile_condition(cond["expr"]) }
                self._compiled[side].append(item)
                if self._context_tf and cond.get("tf") == self._context_tf:
                    self._compiled_context[side].append(item)

            for cond in side_spec.get("exit_conditions_all", []) or []:
                self._compiled_exits[side].append(
                    {"tf": cond["tf"], "expr": cond["expr"], "fn": compile_condition(cond["expr"]) }
                )

        # Stop config per side (nested under each side as 'stop')
        self._stops = {
            "long": (self._spec.get("long") or {}).get("stop") or {},
            "short": (self._spec.get("short") or {}).get("stop") or {},
        }

    def _calculate_stop_loss(self, entry_row, dir_int: int) -> float:
        '''Stop-loss calculation used by the engine during entry.

        The BacktestEngine recalculates stops at execution time via
        `strategy._calculate_stop_loss(...)`, so generated strategies must
        implement this to match the research spec.
        '''
        try:
            entry_price = float(entry_row["close"])
        except Exception:
            return super()._calculate_stop_loss(entry_row, dir_int)

        side = "long" if int(dir_int) == 1 else "short"
        stop_cfg = (self._stops.get(side) or {})
        stop_type = stop_cfg.get("type")

        if stop_type == "percent":
            try:
                pct = float(stop_cfg.get("percent") or 0.0)
            except Exception:
                pct = 0.0
            if pct > 1:
                pct = pct / 100.0
            if pct > 0:
                if side == "long":
                    return entry_price * (1.0 - pct)
                return entry_price * (1.0 + pct)

        if stop_type == "atr":
            atr_len = int(stop_cfg.get("atr_length") or 14)
            try:
                atr_mult = float(stop_cfg.get("atr_multiplier") or 3.0)
            except Exception:
                atr_mult = 3.0
            atr_col = f"atr_{atr_len}"
            atr_val = None
            if hasattr(entry_row, "get"):
                atr_val = entry_row.get(atr_col)
            if atr_val is not None and pd.notna(atr_val) and float(atr_val) > 0:
                if side == "long":
                    return entry_price - atr_mult * float(atr_val)
                return entry_price + atr_mult * float(atr_val)

        if stop_type == "ma":
            ma_type = str(stop_cfg.get("ma_type") or "").strip().lower()
            try:
                ma_len = int(stop_cfg.get("ma_length") or 0)
            except Exception:
                ma_len = 0
            try:
                buf = float(stop_cfg.get("buffer") or 0.0)
            except Exception:
                buf = 0.0

            if ma_type in {"ema", "sma"} and ma_len > 0 and buf >= 0:
                ma_col = f"{ma_type}_close_{ma_len}"
                ma_val = None
                if hasattr(entry_row, "get"):
                    ma_val = entry_row.get(ma_col)
                if ma_val is not None and pd.notna(ma_val):
                    if side == "long":
                        return float(ma_val) - buf
                    return float(ma_val) + buf

        if stop_type == "structure":
            try:
                lookback = int(stop_cfg.get("structure_lookback_bars") or 0)
            except Exception:
                lookback = 0
            try:
                buf = float(stop_cfg.get("buffer") or 0.0)
            except Exception:
                buf = 0.0

            if lookback > 0 and buf >= 0:
                low_col = f"donchian_low_{lookback}"
                high_col = f"donchian_high_{lookback}"
                if hasattr(entry_row, "get"):
                    v = entry_row.get(low_col) if side == "long" else entry_row.get(high_col)
                else:
                    v = None
                if v is not None and pd.notna(v):
                    if side == "long":
                        return float(v) - buf
                    return float(v) + buf

        if stop_type == "candle":
            # Uses precomputed shifted columns (see get_indicators)
            try:
                bars_back = int(stop_cfg.get("candle_bars_back") or 1)
            except Exception:
                bars_back = 1
            try:
                buf = float(stop_cfg.get("buffer") or 0.0)
            except Exception:
                buf = 0.0

            if bars_back > 0 and buf >= 0:
                low_col = f"low_shift_{bars_back}"
                high_col = f"high_shift_{bars_back}"
                if hasattr(entry_row, "get"):
                    v = entry_row.get(low_col) if side == "long" else entry_row.get(high_col)
                else:
                    v = None
                if v is not None and pd.notna(v):
                    if side == "long":
                        return float(v) - buf
                    return float(v) + buf

        # If the research spec didn't define a stop, fall back to StrategyBase.
        return super()._calculate_stop_loss(entry_row, dir_int)

    def get_required_timeframes(self) -> list[str]:
        # Use spec-required timeframes to request engine resampling
        return list(self._spec.get("required_timeframes") or [self.config.timeframes.entry_tf])

    def get_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        tf = tf or "unknown"
        entry_tf = getattr(self.config.timeframes, "entry_tf", None) or self._spec.get("entry_tf")

        # Gather indicator requests for this timeframe from expressions
        requests = []
        for side in ["long", "short"]:
            side_spec = self._spec.get(side)
            if not side_spec or not side_spec.get("enabled", True):
                continue
            for cond in side_spec.get("conditions_all", []):
                if cond.get("tf") != tf:
                    continue
                requests.extend(extract_indicator_requests(cond.get("expr", ""), tf=tf))

            for cond in side_spec.get("exit_conditions_all", []) or []:
                if cond.get("tf") != tf:
                    continue
                requests.extend(extract_indicator_requests(cond.get("expr", ""), tf=tf))

        # Only compute indicators that were requested for this timeframe.
        requests = [r for r in requests if getattr(r, "tf", None) == tf]

        # Report overlays should reflect entry/exit logic only.
        # Trade-management internals (ATR/stop-loss MA/trailing MA) are still computed
        # when needed, but should not automatically clutter the price chart.
        expr_overlay_cols: list[str] = []
        try:
            for r in requests:
                name = getattr(r, "name", None)
                args = getattr(r, "args", ())
                if name in {"ema", "sma"} and len(args) == 2:
                    series_name, length = args
                    expr_overlay_cols.append(f"{name}_{series_name}_{int(length)}")
                elif name in {"donchian_high", "donchian_low"} and len(args) == 1:
                    (length,) = args
                    expr_overlay_cols.append(f"{name}_{int(length)}")
        except Exception:
            expr_overlay_cols = []

        # Include ATR if any side uses ATR stops on this timeframe.
        # (Stops are evaluated on the entry timeframe in generate_signals.)
        if entry_tf and tf == entry_tf:
            for side in ["long", "short"]:
                stop_spec = self._stops.get(side) or {}
                if stop_spec.get("type") == "atr":
                    atr_len = int(stop_spec.get("atr_length") or 14)
                    requests.extend(extract_indicator_requests(f"atr({atr_len})", tf=tf))

                if stop_spec.get("type") == "ma":
                    ma_type = str(stop_spec.get("ma_type") or "").strip().lower()
                    try:
                        ma_len = int(stop_spec.get("ma_length") or 0)
                    except Exception:
                        ma_len = 0
                    if ma_type in {"ema", "sma"} and ma_len > 0:
                        requests.extend(extract_indicator_requests(f"{ma_type}(close, {ma_len})", tf=tf))

                if stop_spec.get("type") == "structure":
                    try:
                        lookback = int(stop_spec.get("structure_lookback_bars") or 0)
                    except Exception:
                        lookback = 0
                    if lookback > 0:
                        # Ensure donchian bands exist for stop calculation
                        requests.extend(extract_indicator_requests(f"donchian_low({lookback})", tf=tf))
                        requests.extend(extract_indicator_requests(f"donchian_high({lookback})", tf=tf))

            # Provide canonical columns used by engine trade management.
            # Stop-loss MA column
            sl_cfg = getattr(self.config, "stop_loss", None)
            if sl_cfg is not None and getattr(sl_cfg, "type", None) in {"SMA", "EMA"}:
                sl_len = int(getattr(sl_cfg, "length", 50))
                ma_name = "sma" if sl_cfg.type == "SMA" else "ema"
                requests.extend(extract_indicator_requests(f"{ma_name}(close, {sl_len})", tf=tf))

            # Trailing stop MA column (engine expects `trail_ema`)
            tr_cfg = getattr(self.config, "trailing_stop", None)
            if tr_cfg is not None and getattr(tr_cfg, "enabled", False) and getattr(tr_cfg, "type", None) in {"EMA", "SMA"}:
                tr_len = int(getattr(tr_cfg, "length", 21))
                ma_name = "ema" if tr_cfg.type == "EMA" else "sma"
                requests.extend(extract_indicator_requests(f"{ma_name}(close, {tr_len})", tf=tf))

        out = ensure_indicators(df, requests)

        # Precompute shifted candle fields used by candle stops (entry timeframe only)
        if entry_tf and tf == entry_tf:
            try:
                need_candle = False
                max_back = 0
                for side in ["long", "short"]:
                    stop_spec = self._stops.get(side) or {}
                    if stop_spec.get("type") != "candle":
                        continue
                    need_candle = True
                    try:
                        b = int(stop_spec.get("candle_bars_back") or 1)
                    except Exception:
                        b = 1
                    if b > max_back:
                        max_back = b
                if need_candle and max_back > 0:
                    for b in range(1, max_back + 1):
                        out[f"low_shift_{b}"] = out["low"].shift(b)
                        out[f"high_shift_{b}"] = out["high"].shift(b)
            except Exception:
                pass

        # Export overlay hints for reporting (entry timeframe only).
        if entry_tf and tf == entry_tf:
            try:
                # Preserve order, drop duplicates.
                seen = set()
                dedup = []
                for c in expr_overlay_cols:
                    if c in seen:
                        continue
                    seen.add(c)
                    dedup.append(c)
                out.attrs["report_overlay_cols"] = dedup
            except Exception:
                pass

        # Create aliases used by the engine.
        if entry_tf and tf == entry_tf:
            sl_cfg = getattr(self.config, "stop_loss", None)
            if sl_cfg is not None and getattr(sl_cfg, "type", None) in {"SMA", "EMA"}:
                sl_len = int(getattr(sl_cfg, "length", 50))
                col = f"{'sma' if sl_cfg.type == 'SMA' else 'ema'}_close_{sl_len}"
                if col in out.columns:
                    out["sl_ma"] = out[col]

            tr_cfg = getattr(self.config, "trailing_stop", None)
            if tr_cfg is not None and getattr(tr_cfg, "enabled", False) and getattr(tr_cfg, "type", None) == "EMA":
                tr_len = int(getattr(tr_cfg, "length", 21))
                col = f"ema_close_{tr_len}"
                if col in out.columns:
                    out["trail_ema"] = out[col]

        return out

    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        entry_tf = getattr(self.config.timeframes, "entry_tf", None) or self._spec.get("entry_tf")
        if entry_tf not in df_by_tf:
            return self._create_signal_dataframe()

        symbol = getattr(getattr(self.config, "market", None), "symbol", None) or "UNKNOWN"

        df_entry = df_by_tf[entry_tf]
        out_rows = []

        # Precompute all condition series once and align them to entry index.
        aligned_cache = {"long": [], "short": []}
        aligned_exit_cache = {"long": [], "short": []}
        aligned_context_cache = {"long": [], "short": []}
        for side in ["long", "short"]:
            compiled = self._compiled.get(side) or []
            for item in compiled:
                try:
                    s = item["fn"](EvalContext(df_by_tf=df_by_tf, tf=item["tf"], target_index=df_entry.index))
                    aligned = s
                except Exception:
                    aligned = None
                aligned_cache[side].append(aligned)

            compiled_ctx = self._compiled_context.get(side) or []
            for item in compiled_ctx:
                try:
                    s = item["fn"](EvalContext(df_by_tf=df_by_tf, tf=item["tf"], target_index=df_entry.index))
                    aligned = s
                except Exception:
                    aligned = None
                aligned_context_cache[side].append(aligned)

            compiled_exits = self._compiled_exits.get(side) or []
            for item in compiled_exits:
                try:
                    s = item["fn"](EvalContext(df_by_tf=df_by_tf, tf=item["tf"], target_index=df_entry.index))
                    aligned = s
                except Exception:
                    aligned = None
                aligned_exit_cache[side].append(aligned)

        # Optional: context invalidation early exit.
        exit_ctx_cfg = None
        try:
            exec_cfg = getattr(self.config, "execution", None)
            exit_ctx_cfg = getattr(exec_cfg, "exit_if_context_invalid", None) if exec_cfg is not None else None
        except Exception:
            exit_ctx_cfg = None

        exit_ctx_enabled = bool(getattr(exit_ctx_cfg, "enabled", False)) if exit_ctx_cfg is not None else False
        exit_ctx_mode = str(getattr(exit_ctx_cfg, "mode", "immediate") or "immediate").strip().lower() if exit_ctx_cfg is not None else "immediate"

        context_fall_edge: dict[str, Optional[pd.Series]] = {"long": None, "short": None}
        if exit_ctx_enabled:
            for side in ["long", "short"]:
                series_list = aligned_context_cache.get(side) or []
                if not series_list:
                    continue

                ok_series = None
                for s in series_list:
                    if s is None:
                        ok_series = None
                        break
                    ss = s.reindex(df_entry.index)
                    ss = ss.fillna(False).astype(bool)
                    ok_series = ss if ok_series is None else (ok_series & ss)

                if ok_series is None:
                    continue

                prev = ok_series.shift(1)
                if len(prev) > 0:
                    prev.iloc[0] = ok_series.iloc[0]
                edge = (prev.astype(bool)) & (~ok_series.astype(bool))
                context_fall_edge[side] = edge

        # Evaluate at entry timeframe timestamps only (engine will fill next bar open)
        for ts in df_entry.index:
            # Early blackout block
            if self._should_block_signal_generation(ts, symbol):
                continue

            for side in ["long", "short"]:
                compiled = self._compiled.get(side) or []
                if not compiled:
                    continue

                ok = True
                for idx, _ in enumerate(compiled):
                    aligned = aligned_cache[side][idx]
                    if aligned is None:
                        ok = False
                        break
                    try:
                        passed = bool(aligned.loc[ts])
                    except Exception:
                        passed = False
                    if not passed:
                        ok = False
                        break

                if not ok:
                    continue

                # Build signal
                entry_price = float(df_entry.loc[ts, "close"])

                stop_price = None
                stop_cfg = self._stops.get(side, {})
                stop_type = stop_cfg.get("type")
                if stop_type == "percent":
                    pct = float(stop_cfg.get("percent") or 0.0)
                    if pct > 0:
                        if side == "long":
                            stop_price = entry_price * (1 - pct)
                        else:
                            stop_price = entry_price * (1 + pct)
                elif stop_type == "atr":
                    atr_len = int(stop_cfg.get("atr_length") or 14)
                    atr_mult = float(stop_cfg.get("atr_multiplier") or 3.0)
                    atr_col = f"atr_{atr_len}"
                    atr_val = None
                    if atr_col in df_entry.columns:
                        atr_val = df_entry.loc[ts, atr_col]
                    if atr_val is not None and pd.notna(atr_val) and float(atr_val) > 0:
                        if side == "long":
                            stop_price = entry_price - atr_mult * float(atr_val)
                        else:
                            stop_price = entry_price + atr_mult * float(atr_val)

                if stop_price is None:
                    # Conservative fallback
                    stop_price = entry_price * (0.99 if side == "long" else 1.01)

                sig = {
                    "timestamp": ts,
                    "action": "entry",
                    "immediate_execution": True,
                    "signal_time": ts,
                    "direction": side,
                    "entry_price": entry_price,
                    "stop_price": float(stop_price),
                    "weight": 1.0,
                    "metadata": {"generated": True, "spec_name": self._spec.get("name")},
                }

                # Apply filter chain
                if not self.apply_filters(pd.Series(sig), ts, symbol, df_by_tf):
                    continue

                out_rows.append(sig)

            # Exit rules: emit exit intent at bar close (engine will close at current price)
            for side in ["long", "short"]:
                compiled_exits = self._compiled_exits.get(side) or []
                if not compiled_exits:
                    continue

                ok = True
                for idx, _ in enumerate(compiled_exits):
                    aligned = aligned_exit_cache[side][idx]
                    if aligned is None:
                        ok = False
                        break
                    try:
                        passed = bool(aligned.loc[ts])
                    except Exception:
                        passed = False
                    if not passed:
                        ok = False
                        break

                if not ok:
                    continue

                out_rows.append(
                    {
                        "timestamp": ts,
                        "action": "exit",
                        "direction": side,
                        "exit_reason": "signal_exit",
                        "metadata": {
                            "generated": True,
                            "spec_name": self._spec.get("name"),
                            "exit": True,
                        },
                    }
                )

            # Context invalidation exit/tighten intent (optional)
            if exit_ctx_enabled:
                for side in ["long", "short"]:
                    edge = context_fall_edge.get(side)
                    if edge is None:
                        continue
                    try:
                        should_fire = bool(edge.loc[ts])
                    except Exception:
                        should_fire = False
                    if not should_fire:
                        continue

                    if exit_ctx_mode == "tighten_stop":
                        out_rows.append(
                            {
                                "timestamp": ts,
                                "action": "tighten_stop",
                                "direction": side,
                                "tighten_mode": "breakeven",
                                "metadata": {
                                    "generated": True,
                                    "spec_name": self._spec.get("name"),
                                    "context_invalid": True,
                                    "mode": "tighten_stop",
                                },
                            }
                        )
                    else:
                        out_rows.append(
                            {
                                "timestamp": ts,
                                "action": "exit",
                                "direction": side,
                                "exit_reason": "context_invalid",
                                "metadata": {
                                    "generated": True,
                                    "spec_name": self._spec.get("name"),
                                    "context_invalid": True,
                                    "mode": "immediate",
                                },
                            }
                        )

        if not out_rows:
            return self._create_signal_dataframe()

        df_sig = pd.DataFrame(out_rows).set_index("timestamp")
        return df_sig
"""


class StrategyCompiler:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)

    def load_spec(self, spec_path: Path) -> StrategySpec:
        spec_path = Path(spec_path)
        with open(spec_path, "r") as f:
            data = yaml.safe_load(f)
        return StrategySpec.model_validate(data)

    def compile_to_folder(self, spec: StrategySpec, out_dir: Path | None = None) -> Path:
        out_dir = Path(out_dir) if out_dir else (self.repo_root / "strategies" / spec.name)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Generate config.yml in standard TradingLab format (minimal viable)
        # StrategyConfig requires `moving_averages` and `alignment_rules`.
        config = {
            "strategy_name": spec.name,
            "description": spec.description,
            "market": {
                "symbol": spec.market.symbol,
                "exchange": spec.market.exchange or "unknown",
                "market_type": spec.market.market_type or "spot",
                "base_timeframe": "1m",
                "leverage": 1.0,
            },
            "timeframes": {
                "signal_tf": spec.entry_tf,
                "entry_tf": spec.entry_tf,
            },
            "moving_averages": {
                "ma_fast": {"enabled": True, "length": 20},
                "ma_slow": {"enabled": True, "length": 50},
            },
            "alignment_rules": {
                "long": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
                "short": {"macd_bars_signal_tf": 0, "macd_bars_entry_tf": 0},
            },
            "risk": {
                "risk_per_trade_pct": float(spec.risk_per_trade_pct),
                "sizing_mode": str(getattr(spec, "sizing_mode", "account_size")),
                "account_size": float(getattr(spec, "account_size", 10000.0)),
            },
            "trade_limits": {
                "max_trades_per_day": int(getattr(spec, "max_trades_per_day", 1)),
                "max_daily_loss_pct": getattr(spec, "max_daily_loss_pct", 3.0),
            },
            "execution": {},
            "filters": {},
            # Filters: these keys match how FilterManager merges strategy config
            "calendar_filters": spec.filters.calendar_filters or {},
            "regime_filters": spec.filters.regime_filters or {},
            "news_filter": spec.filters.news_filter or {},
            "volume_filters": spec.filters.volume_filters or {},
        }

        # Execution overrides (optional)
        exec_spec = getattr(spec, "execution", None)
        if exec_spec is not None:
            exec_dict = exec_spec.model_dump(exclude_none=True)
            if exec_dict:
                config["execution"].update(exec_dict)

        # Trade management overrides (merged with config/master_trade_management.yml by engine)
        tm = getattr(spec, "trade_management", None)
        if tm is not None:
            if tm.trailing_stop is not None:
                config["trailing_stop"] = tm.trailing_stop.model_dump(exclude_none=True)

            if tm.take_profit is not None:
                tp_dict: dict[str, Any] = {
                    "levels": [
                        {
                            "enabled": lvl.enabled,
                            "type": "r_based",
                            "target_r": float(lvl.target_r),
                            "exit_pct": float(lvl.exit_pct),
                        }
                        for lvl in (tm.take_profit.levels or [])
                    ]
                }
                if tm.take_profit.enabled is not None:
                    tp_dict["enabled"] = bool(tm.take_profit.enabled)
                config["take_profit"] = tp_dict

            if tm.partial_exit is not None:
                pe_dict: dict[str, Any] = {
                    "levels": [
                        {
                            "enabled": lvl.enabled,
                            "type": "r_based",
                            "level_r": float(lvl.level_r),
                            "exit_pct": float(lvl.exit_pct),
                        }
                        for lvl in (tm.partial_exit.levels or [])
                    ]
                }
                if tm.partial_exit.enabled is not None:
                    pe_dict["enabled"] = bool(tm.partial_exit.enabled)
                config["partial_exit"] = pe_dict

        # Backtest costs: only include when the spec explicitly overrides.
        # If omitted, market profile defaults (config/market_profiles.yml) apply.
        try:
            commissions = getattr(spec, "commissions", None)
        except Exception:
            commissions = None
        try:
            slippage_ticks = getattr(spec, "slippage_ticks", None)
        except Exception:
            slippage_ticks = None

        bt: dict[str, Any] = {}
        if commissions is not None:
            bt["commissions"] = float(commissions)
        if slippage_ticks is not None:
            bt["slippage_ticks"] = float(slippage_ticks)
        if bt:
            config["backtest"] = bt

        # Ensure it validates as a StrategyConfig (best effort)
        validate_strategy_config(config)

        config_path = out_dir / "config.yml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # 2) Generate strategy.py (wrapper used by run_backtest discovery)
        spec_summary = spec.summary()

        # Include any timeframes referenced inside expressions via at()/@ syntax.
        discovered_tfs: set[str] = set()
        for side in [spec.long, spec.short]:
            if not side:
                continue
            for cond in side.conditions_all:
                discovered_tfs |= extract_timeframe_refs(cond.expr, default_tf=cond.tf)
            for cond in side.exit_conditions_all:
                discovered_tfs |= extract_timeframe_refs(cond.expr, default_tf=cond.tf)

        required_tfs = sorted(set(spec.required_timeframes()) | discovered_tfs)
        # Keep entry_tf first for stable UX.
        if spec.entry_tf in required_tfs:
            required_tfs = [spec.entry_tf] + [tf for tf in required_tfs if tf != spec.entry_tf]
        strategy_py = out_dir / "strategy.py"
        with open(strategy_py, "w") as f:
            spec_payload = {
                "name": spec_summary["name"],
                "description": spec_summary["description"],
                "market": spec_summary["market"],
                "entry_tf": spec_summary["entry_tf"],
                "context_tf": spec_summary.get("context_tf"),
                "required_timeframes": required_tfs,
                "long": spec_summary["long"],
                "short": spec_summary["short"],
            }
            spec_json_str = json.dumps(spec_payload, sort_keys=True)
            spec_json_literal = json.dumps(spec_json_str)
            f.write(_STRATEGY_TEMPLATE.replace("__SPEC_JSON__", spec_json_literal))

        # 3) Provide a thin top-level export for discovery (consistent with existing patterns)
        init_py = out_dir / "__init__.py"
        if not init_py.exists():
            init_py.write_text("from .strategy import GeneratedResearchStrategy\n\n__all__ = ['GeneratedResearchStrategy']\n")

        # 4) README
        readme = out_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {spec.name}\n\n" +
                "This strategy was generated by the TradingLab research layer from a YAML spec.\n\n"
                "Re-generate from spec instead of editing code directly.\n"
            )

        return out_dir
