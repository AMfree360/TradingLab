from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


TimeframeRef = str  # e.g. '15m', '1h', '4h', '1d'


class MarketSpecLite(BaseModel):
    symbol: str = Field(..., description="Market symbol, e.g. BTCUSDT, EURUSD")
    exchange: Optional[str] = Field(None, description="Optional exchange identifier")
    market_type: Optional[Literal["spot", "futures"]] = None


class ConditionSpec(BaseModel):
    tf: TimeframeRef = Field(..., description="Timeframe to evaluate this condition on")
    expr: str = Field(..., description="Boolean expression in the research DSL")


class SideRuleSpec(BaseModel):
    enabled: bool = True
    conditions_all: list[ConditionSpec] = Field(default_factory=list, description="All conditions must be true")

    # Optional exit rules (evaluated on bar close; engine closes at current price)
    exit_conditions_all: list[ConditionSpec] = Field(
        default_factory=list,
        description="All exit conditions must be true to trigger a rule-based exit",
    )

    # Optional stop definition (kept simple at research layer, implemented by generated strategy)
    stop_type: Optional[Literal["percent", "atr", "ma", "structure", "candle"]] = None

    # percent
    stop_percent: Optional[float] = None

    # atr
    atr_length: Optional[int] = None
    atr_multiplier: Optional[float] = None

    # ma (absolute buffer in instrument units)
    stop_ma_type: Optional[Literal["ema", "sma"]] = None
    stop_ma_length: Optional[int] = None
    stop_buffer: Optional[float] = None

    # structure: stop at Donchian high/low over lookback_bars (+/- absolute buffer)
    stop_structure_lookback_bars: Optional[int] = None

    # candle: stop at prior candle high/low (+/- absolute buffer)
    stop_candle_bars_back: Optional[int] = None

    @model_validator(mode="after")
    def _validate_stop(self) -> "SideRuleSpec":
        if self.stop_type is None:
            return self
        if self.stop_type == "percent":
            if self.stop_percent is None or self.stop_percent <= 0:
                raise ValueError("stop_percent must be > 0 when stop_type=percent")
        if self.stop_type == "atr":
            if self.atr_length is None or self.atr_length <= 0:
                raise ValueError("atr_length must be > 0 when stop_type=atr")
            if self.atr_multiplier is None or self.atr_multiplier <= 0:
                raise ValueError("atr_multiplier must be > 0 when stop_type=atr")
        if self.stop_type == "ma":
            if self.stop_ma_type not in {"ema", "sma"}:
                raise ValueError("stop_ma_type must be 'ema' or 'sma' when stop_type=ma")
            if self.stop_ma_length is None or self.stop_ma_length <= 0:
                raise ValueError("stop_ma_length must be > 0 when stop_type=ma")
            if self.stop_buffer is None or self.stop_buffer < 0:
                raise ValueError("stop_buffer must be >= 0 when stop_type=ma")
        if self.stop_type == "structure":
            if self.stop_structure_lookback_bars is None or self.stop_structure_lookback_bars <= 0:
                raise ValueError("stop_structure_lookback_bars must be > 0 when stop_type=structure")
            if self.stop_buffer is None or self.stop_buffer < 0:
                raise ValueError("stop_buffer must be >= 0 when stop_type=structure")
        if self.stop_type == "candle":
            if self.stop_candle_bars_back is None or self.stop_candle_bars_back <= 0:
                raise ValueError("stop_candle_bars_back must be > 0 when stop_type=candle")
            if self.stop_buffer is None or self.stop_buffer < 0:
                raise ValueError("stop_buffer must be >= 0 when stop_type=candle")
        return self


class FiltersSpec(BaseModel):
    """Pass-through filter config.

    This is intentionally permissive: the research layer should let users select
    filters in a human-readable way, but the underlying config is still the
    canonical TradingLab filter schema.
    """

    calendar_filters: dict[str, Any] = Field(default_factory=dict)
    regime_filters: dict[str, Any] = Field(default_factory=dict)
    news_filter: dict[str, Any] = Field(default_factory=dict)
    volume_filters: dict[str, Any] = Field(default_factory=dict)


class TrailingStopSpec(BaseModel):
    """User-friendly trailing stop config.

    This maps directly onto TradingLab's `TrailingStopConfig` keys.
    """

    enabled: bool = False
    type: Literal["EMA", "SMA", "ATR", "percentage", "fixed_distance"] = "EMA"
    length: int = Field(default=21, gt=0, description="MA length for EMA/SMA trailing")
    activation_type: Literal["r_based", "atr_based", "price_based", "time_based"] = "r_based"
    activation_r: float = Field(default=0.5, ge=0.0)
    stepped: bool = True
    min_move_pips: float = Field(default=7.0, ge=0.0)


class TakeProfitLevelSpec(BaseModel):
    enabled: bool = True
    target_r: float = Field(..., gt=0.0, description="Target R multiple")
    exit_pct: float = Field(default=100.0, ge=0.0, le=100.0)


class TakeProfitSpec(BaseModel):
    """User-friendly take profit.

    Currently compiles to `type: r_based` levels.
    """

    enabled: Optional[bool] = Field(
        default=None,
        description="If set, overrides master take_profit.enabled for this strategy",
    )
    levels: list[TakeProfitLevelSpec] = Field(default_factory=list)


class PartialExitLevelSpec(BaseModel):
    enabled: bool = True
    level_r: float = Field(..., gt=0.0, description="Partial-exit R multiple")
    exit_pct: float = Field(default=50.0, ge=0.0, le=100.0)


class PartialExitSpec(BaseModel):
    enabled: Optional[bool] = Field(
        default=None,
        description="If set, overrides master partial_exit.enabled for this strategy",
    )
    levels: list[PartialExitLevelSpec] = Field(default_factory=list)


class TradeManagementSpec(BaseModel):
    """Trade management overrides.

    These are emitted into the generated `config.yml` so the engine's
    TradeManagementManager can merge them with `config/master_trade_management.yml`.
    """

    trailing_stop: Optional[TrailingStopSpec] = None
    take_profit: Optional[TakeProfitSpec] = None
    partial_exit: Optional[PartialExitSpec] = None


class ExitIfContextInvalidSpec(BaseModel):
    enabled: bool = False
    mode: Literal["immediate", "tighten_stop"] = "immediate"


class ExecutionSpec(BaseModel):
    """User-friendly execution/session controls.

    This compiles into TradingLab's `execution` config section.
    """

    flatten_enabled: Optional[bool] = Field(
        default=None,
        description="If set, overrides config.execution.flatten_enabled",
    )
    flatten_time: Optional[str] = Field(
        default=None,
        description="UTC time 'HH:MM' to flatten all positions (when enabled)",
    )
    stop_signal_search_minutes_before: Optional[int] = Field(
        default=None,
        description="Stop searching for new signals N minutes before flatten_time",
    )
    use_intraday_margin: Optional[bool] = Field(
        default=None,
        description="For futures: if True use intraday margin; if False use overnight initial margin",
    )
    exit_if_context_invalid: Optional[ExitIfContextInvalidSpec] = Field(
        default=None,
        description="Optional early-exit behavior when context becomes invalid (default: disabled).",
    )


class StrategySpec(BaseModel):
    """Human-readable strategy specification.

    This is meant to be serialized as YAML/JSON, reviewed, then compiled into a
    standard TradingLab strategy folder.
    """

    name: str = Field(..., description="Strategy name (folder name)")
    description: str = Field("", description="Human description")

    market: MarketSpecLite

    # Core timeframes
    entry_tf: TimeframeRef = Field(..., description="Execution/reference timeframe")
    # Optional additional timeframes for confirmation/context
    extra_tfs: list[TimeframeRef] = Field(default_factory=list)

    # Optional context timeframe (primarily for guided builders / clarity in specs)
    context_tf: Optional[TimeframeRef] = Field(
        default=None,
        description="Optional context timeframe used for regime/filters; if omitted, context is evaluated on entry_tf",
    )

    long: Optional[SideRuleSpec] = None
    short: Optional[SideRuleSpec] = None

    filters: FiltersSpec = Field(default_factory=FiltersSpec)

    trade_management: TradeManagementSpec = Field(default_factory=TradeManagementSpec)

    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)

    # Output config defaults (can be overridden later by config profiles)
    risk_per_trade_pct: float = 1.0

    # Risk sizing
    sizing_mode: Literal["equity", "account_size", "daily_equity"] = Field(
        default="account_size",
        description="Position sizing base: 'account_size' (fixed), 'equity' (compounds every trade), or 'daily_equity' (compounds once per day).",
    )
    account_size: float = Field(
        default=10000.0,
        gt=0.0,
        description="Initial account size for position sizing (used when sizing_mode='account_size').",
    )

    # Daily risk controls
    max_trades_per_day: int = Field(
        default=1,
        ge=1,
        description="Stop taking NEW entries for the day after this many entries.",
    )
    max_daily_loss_pct: Optional[float] = Field(
        default=3.0,
        gt=0.0,
        le=100.0,
        description="Stop taking NEW entries for the day once realized P&L after costs is <= -X% of the day base. Set to null to disable.",
    )

    # Backtest execution-cost overrides (when set). If omitted/null, market profile defaults apply.
    commissions: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Commission override rate (e.g. 0.0004 = 0.04%). Null = use market defaults.",
    )
    slippage_ticks: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Slippage override in native units (pips for FX, ticks for futures, otherwise price). Null = use market defaults.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "StrategySpec":
        if not self.name or not self.name.strip():
            raise ValueError("name is required")
        if self.long is None and self.short is None:
            raise ValueError("At least one of long/short must be provided")

        used_tfs = {self.entry_tf}
        used_tfs.update(self.extra_tfs)
        if self.context_tf:
            used_tfs.add(self.context_tf)
        for side in [self.long, self.short]:
            if not side:
                continue
            for cond in side.conditions_all:
                used_tfs.add(cond.tf)
        if self.entry_tf not in used_tfs:
            raise ValueError("entry_tf must be included in used timeframes")
        return self

    def required_timeframes(self) -> list[str]:
        used = {self.entry_tf}
        used.update(self.extra_tfs)
        if self.context_tf:
            used.add(self.context_tf)
        for side in [self.long, self.short]:
            if not side:
                continue
            for cond in side.conditions_all:
                used.add(cond.tf)
        # Keep stable order: entry_tf first, then the rest sorted
        rest = sorted([tf for tf in used if tf != self.entry_tf])
        return [self.entry_tf] + rest

    def summary(self) -> dict[str, Any]:
        def _side_summary(side: Optional[SideRuleSpec]) -> dict[str, Any] | None:
            if side is None:
                return None
            return {
                "enabled": side.enabled,
                "conditions_all": [{"tf": c.tf, "expr": c.expr} for c in side.conditions_all],
                "exit_conditions_all": [{"tf": c.tf, "expr": c.expr} for c in side.exit_conditions_all],
                "stop": {
                    "type": side.stop_type,
                    "percent": side.stop_percent,
                    "atr_length": side.atr_length,
                    "atr_multiplier": side.atr_multiplier,
                    "ma_type": side.stop_ma_type,
                    "ma_length": side.stop_ma_length,
                    "buffer": side.stop_buffer,
                    "structure_lookback_bars": side.stop_structure_lookback_bars,
                    "candle_bars_back": side.stop_candle_bars_back,
                },
            }

        return {
            "name": self.name,
            "description": self.description,
            "market": self.market.model_dump(),
            "entry_tf": self.entry_tf,
            "context_tf": self.context_tf,
            "required_timeframes": self.required_timeframes(),
            "long": _side_summary(self.long),
            "short": _side_summary(self.short),
            "filters": self.filters.model_dump(),
            "trade_management": self.trade_management.model_dump(),
            "execution": self.execution.model_dump(),
            "risk_per_trade_pct": self.risk_per_trade_pct,
        }
