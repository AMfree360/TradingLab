"""Configuration validation schemas using Pydantic."""

from typing import Literal, Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import yaml
from pathlib import Path


class MarketConfig(BaseModel):
    """Market configuration."""
    exchange: str
    symbol: str
    base_timeframe: str = "1m"
    market_type: Literal["spot", "futures"] = "spot"  # Spot or futures trading
    leverage: float = Field(default=1.0, ge=1.0, le=125.0)  # Leverage (1.0 = no leverage, max 125x for Binance)


class TimeframeConfig(BaseModel):
    """Timeframe configuration."""
    signal_tf: str
    entry_tf: str


class MovingAverageConfig(BaseModel):
    """Moving average configuration."""
    enabled: bool = True
    length: int = Field(gt=0)


class MACDConfig(BaseModel):
    """MACD indicator configuration."""
    fast: int = Field(default=15, gt=0)
    slow: int = Field(default=30, gt=0)
    signal: int = Field(default=9, gt=0)


class MACDExtendedConfig(MACDConfig):
    """MACD configuration with minimum histogram bars requirement."""
    min_bars: int = Field(default=1, ge=0)


class MAGroupConfig(BaseModel):
    """Moving average group configuration for a timeframe."""
    fast: int = Field(default=5, gt=0)
    medium: int = Field(default=15, gt=0)
    slow: int = Field(default=30, gt=0)
    slowest: int = Field(default=50, gt=0)
    optional: int = Field(default=100, ge=0)
    use_fast: bool = True
    use_medium: bool = True
    use_slow: bool = True
    use_slowest: bool = True
    use_optional: bool = True
    early_misalignment: Optional[int] = Field(
        default=None,
        description="EMA period inserted between two enabled MAs when only two are active. Use 0 to disable.",
        ge=0,  # Allow 0 to disable (matches NinjaTrader behavior)
    )


class AlignmentRulesConfig(BaseModel):
    """Alignment rules for long/short signals."""
    macd_bars_signal_tf: int = Field(ge=0)
    macd_bars_entry_tf: int = Field(ge=0)


class MinStopDistanceConfig(BaseModel):
    """Minimum stop distance configuration using ATR multiplier or fixed distance."""
    enabled: bool = False
    type: Literal["atr", "fixed"] = Field(
        default="atr",
        description="Type of minimum stop distance: 'atr' (ATR-based) or 'fixed' (fixed pips/points/price)"
    )
    # ATR-based settings
    atr_multiplier: float = Field(default=1.5, gt=0.0, description="ATR multiplier for minimum stop distance (e.g., 1.5 = 1.5x ATR)")
    atr_period: int = Field(default=14, gt=0, description="ATR period for calculation")
    # Fixed distance settings
    fixed_distance_pips: float = Field(default=6.0, gt=0.0, description="Fixed minimum stop distance in pips/points/price (default: 6.0)")
    mode: Literal["skip", "use_atr", "use_fixed"] = Field(
        default="skip",
        description="What to do if calculated stop < minimum: 'skip' (reject trade), 'use_atr' (use ATR-based stop), or 'use_fixed' (use fixed distance stop)"
    )


class StopLossConfig(BaseModel):
    """Stop loss configuration."""
    type: Literal["SMA", "EMA"] = "SMA"
    length: int = Field(gt=0, default=50)
    buffer_pips: float = Field(default=1.0, ge=0.0, description="Stop loss buffer in pips/price units (default: 1 pip for EURUSD)")
    buffer_unit: Literal["pip", "price", "points"] = Field(default="pip", description="Unit for buffer: 'pip' (for forex), 'price' (for crypto/stocks), or 'points' (for futures)")
    pip_size: float = Field(default=0.0001, gt=0.0, description="Price value of one pip for pip-based buffers (e.g., 0.0001 for most FX)")
    min_stop_distance: MinStopDistanceConfig = Field(default_factory=MinStopDistanceConfig)


class TrailingStopTypeConfig(BaseModel):
    """Configuration for a specific trailing stop type."""
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, description="Weight when combining multiple trailing stop types")


class TrailingStopConfig(BaseModel):
    """Trailing stop configuration with support for multiple types."""
    enabled: bool = False
    type: Literal["EMA", "SMA", "ATR", "percentage", "fixed_distance"] = Field(
        default="EMA",
        description="Primary trailing stop type"
    )
    # EMA/SMA specific
    length: int = Field(default=21, gt=0)
    # ATR specific
    atr_period: int = Field(default=14, gt=0)
    atr_multiplier: float = Field(default=1.5, gt=0.0)
    # Percentage specific
    percentage: float = Field(default=2.0, gt=0.0, description="Percentage trailing (e.g., 2.0 = 2% trailing)")
    # Fixed distance specific
    fixed_distance_pips: float = Field(default=10.0, ge=0.0, description="Fixed distance in pips/points")
    # Activation
    activation_type: Literal["r_based", "atr_based", "price_based", "time_based"] = Field(
        default="r_based",
        description="How trailing activates"
    )
    activation_r: float = Field(default=0.5, ge=0.0, description="Activation R multiple (for r_based)")
    activation_atr_multiplier: float = Field(default=1.0, gt=0.0, description="Activation ATR multiplier (for atr_based)")
    activation_price_pips: float = Field(default=20.0, ge=0.0, description="Activation price distance in pips (for price_based)")
    activation_time_minutes: int = Field(default=60, gt=0, description="Activation time in minutes (for time_based)")
    # Stepped trailing
    stepped: bool = Field(default=True, description="Use stepped R-based trailing like TitanEA")
    min_move_pips: float = Field(default=7.0, ge=0.0, description="Minimum price movement in pips before moving SL")
    min_move_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-instrument overrides for trailing min move (symbol substring -> pips/points)",
    )


class PartialExitLevelConfig(BaseModel):
    """Configuration for a single partial exit level."""
    enabled: bool = True
    type: Literal["r_based", "atr_based", "price_based", "percentage_based", "time_based"] = Field(
        default="r_based",
        description="How partial exit level is calculated"
    )
    level_r: Optional[float] = Field(default=None, gt=0.0, description="R multiple for r_based (e.g., 1.5 = 1.5R)")
    level_atr_multiplier: Optional[float] = Field(default=None, gt=0.0, description="ATR multiplier for atr_based")
    level_price_pips: Optional[float] = Field(default=None, ge=0.0, description="Price distance in pips for price_based")
    level_percentage: Optional[float] = Field(default=None, gt=0.0, description="Percentage profit for percentage_based (e.g., 5.0 = 5%)")
    level_time_minutes: Optional[int] = Field(default=None, gt=0, description="Time in minutes for time_based")
    exit_pct: float = Field(default=50.0, ge=0.0, le=100.0, description="Percentage of position to exit at this level")


class PartialExitConfig(BaseModel):
    """Partial exit configuration with support for multiple levels."""
    enabled: bool = False
    levels: List[PartialExitLevelConfig] = Field(
        default_factory=list,
        description="List of partial exit levels (executed in order)"
    )
    # Legacy support - single level (converted to levels in validator)
    level_r: Optional[float] = Field(default=None, gt=0.0, description="Legacy: single R-based level (converted to levels if used)")
    exit_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Legacy: single exit percentage (converted to levels if used)")
    
    @model_validator(mode='after')
    def convert_legacy_fields(self):
        """Convert legacy level_r/exit_pct to levels list if needed."""
        # If levels is empty and legacy fields exist, create level from legacy
        if not self.levels and self.level_r is not None:
            self.levels = [PartialExitLevelConfig(
                level_r=self.level_r,
                exit_pct=self.exit_pct or 80.0
            )]
        return self


class TakeProfitLevelConfig(BaseModel):
    """Configuration for a single take profit level."""
    enabled: bool = True
    type: Literal["r_based", "atr_based", "price_based", "percentage_based", "time_based"] = Field(
        default="r_based",
        description="How take profit level is calculated"
    )
    target_r: Optional[float] = Field(default=None, gt=0.0, description="R multiple for r_based (e.g., 3.0 = 3R)")
    target_atr_multiplier: Optional[float] = Field(default=None, gt=0.0, description="ATR multiplier for atr_based")
    target_price_pips: Optional[float] = Field(default=None, ge=0.0, description="Price distance in pips for price_based")
    target_percentage: Optional[float] = Field(default=None, gt=0.0, description="Percentage profit for percentage_based (e.g., 10.0 = 10%)")
    target_time_minutes: Optional[int] = Field(default=None, gt=0, description="Time in minutes for time_based")
    exit_pct: float = Field(default=100.0, ge=0.0, le=100.0, description="Percentage of position to exit at this level (100 = full exit)")


class TakeProfitConfig(BaseModel):
    """Take profit configuration with support for multiple levels."""
    enabled: bool = True
    levels: List[TakeProfitLevelConfig] = Field(
        default_factory=list,
        description="List of take profit levels (executed in order)"
    )
    # Legacy support - single level (converted to levels in validator)
    target_r: Optional[float] = Field(default=None, gt=0.0, description="Legacy: single R-based target (converted to levels if used)")
    
    @model_validator(mode='after')
    def convert_legacy_fields(self):
        """Convert legacy target_r to levels list if needed."""
        # If levels is empty and legacy field exists, create level from legacy
        if not self.levels and self.target_r is not None:
            self.levels = [TakeProfitLevelConfig(
                target_r=self.target_r,
                exit_pct=100.0
            )]
        # Default if nothing specified
        if not self.levels:
            self.levels = [TakeProfitLevelConfig(target_r=3.0, exit_pct=100.0)]
        return self


class SessionSlotConfig(BaseModel):
    """Individual session slot configuration."""
    enabled: bool = True
    start: str  # Format: "HH:MM"
    end: str    # Format: "HH:MM"

class SessionFilterConfig(BaseModel):
    """Session filter configuration."""
    enabled: bool = False
    slots: List[Union[List[str], Dict[str, Any]]] = Field(
        default_factory=lambda: [["00:00", "08:00"], ["12:00", "16:00"], ["20:00", "22:00"]],
        description="List of session slots. Can be list of [start, end] strings or dict with enabled, start, end keys"
    )
    
    @field_validator('slots')
    @classmethod
    def validate_slots(cls, v: List[Union[List[str], Dict[str, Any]]]) -> List[Union[List[str], Dict[str, Any]]]:
        """Normalize slots to handle both list and dict formats."""
        normalized = []
        for slot in v:
            if isinstance(slot, list) and len(slot) == 2:
                # Convert [start, end] to dict format
                normalized.append({
                    'enabled': True,
                    'start': slot[0],
                    'end': slot[1]
                })
            elif isinstance(slot, dict):
                # Already in dict format, ensure it has required keys
                if 'start' in slot and 'end' in slot:
                    normalized.append({
                        'enabled': slot.get('enabled', True),
                        'start': slot['start'],
                        'end': slot['end']
                    })
        return normalized


class TrendFilterConfig(BaseModel):
    """Trend filter configuration."""
    enabled: bool = False
    length: Optional[int] = None
    min_slope: Optional[float] = None
    min_sep_pct: Optional[float] = None


class FiltersConfig(BaseModel):
    """All filter configurations."""
    session_filters: SessionFilterConfig = Field(default_factory=SessionFilterConfig)
    sma_slope: TrendFilterConfig = Field(default_factory=TrendFilterConfig)
    ema_separation: TrendFilterConfig = Field(default_factory=TrendFilterConfig)
    higher_highs_lows: TrendFilterConfig = Field(default_factory=TrendFilterConfig)


class DayOfWeekFilterConfig(BaseModel):
    """Day-of-week calendar filter."""
    enabled: bool = False
    allowed_days: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4], description="0=Monday ... 6=Sunday")


class MonthOfYearFilterConfig(BaseModel):
    """Month-of-year calendar filter."""
    enabled: bool = False
    allowed_months: List[int] = Field(default_factory=lambda: list(range(1, 13)))


class TimeWindowConfig(BaseModel):
    """Time window with optional symbol filter."""
    start: str
    end: str
    days: List[str] = Field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri"])
    symbols: List[str] = Field(default_factory=list)
    enabled: bool = True


class TradingSessionConfig(BaseModel):
    """Trading session configuration with UTC+00 time range."""
    enabled: bool = True
    start: str  # Format: "HH:MM" in UTC+00
    end: str    # Format: "HH:MM" in UTC+00 (can span midnight, e.g., 23:00-08:00)


class CalendarFiltersConfig(BaseModel):
    """Calendar and session filters mirroring TitanEA options."""
    master_filters_enabled: bool = False
    day_of_week: DayOfWeekFilterConfig = Field(default_factory=DayOfWeekFilterConfig)
    month_of_year: MonthOfYearFilterConfig = Field(default_factory=MonthOfYearFilterConfig)
    trading_sessions_enabled: bool = Field(
        default=False,
        description="Enable TradingSessionFilter. When False, trading_sessions definitions are ignored for filtering.",
    )
    trading_sessions: Dict[str, TradingSessionConfig] = Field(
        default_factory=lambda: {
            "Asia": TradingSessionConfig(enabled=False, start="23:00", end="08:00"),
            "London": TradingSessionConfig(enabled=False, start="07:00", end="16:00"),
            "NewYork": TradingSessionConfig(enabled=False, start="13:00", end="21:00"),
        },
        description="Trading sessions with UTC+00 time ranges. Each session can be enabled/disabled."
    )
    # Legacy field for backward compatibility - will be ignored if trading_sessions is present
    allowed_sessions: List[str] = Field(default_factory=lambda: [], description="Legacy: use trading_sessions instead")
    instrument_sessions: Dict[str, List[str]] = Field(default_factory=dict)
    time_blackouts_enabled: bool = False
    blackout_windows: List[TimeWindowConfig] = Field(default_factory=list)
    news_filter_enabled: bool = False
    
    @field_validator('trading_sessions')
    @classmethod
    def validate_trading_sessions(cls, v: Dict[str, Any]) -> Dict[str, TradingSessionConfig]:
        """Convert dict values to TradingSessionConfig objects."""
        if not isinstance(v, dict):
            return v
        return {
            k: TradingSessionConfig(**val) if isinstance(val, dict) else val
            for k, val in v.items()
        }


class NewsEventConfig(BaseModel):
    """Custom news/event blackout."""
    name: str
    time: str  # ISO datetime string
    buffer_before: int = Field(default=15, ge=0)
    buffer_after: int = Field(default=15, ge=0)
    symbols: List[str] = Field(default_factory=list)


class NewsFilterConfig(BaseModel):
    """News blackout configuration."""
    enabled: bool = False
    events: List[NewsEventConfig] = Field(default_factory=list)


class RiskConfig(BaseModel):
    """Risk management configuration.
    
    sizing_mode options:
    - "account_size": Fixed risk based on initial capital (more realistic, recommended for testing)
    - "equity": Dynamic risk based on current equity (compounds gains, can lead to unrealistic results)
    """
    sizing_mode: Literal["equity", "account_size", "daily_equity"] = Field(
        default="account_size",
        description=(
            "Position sizing base: "
            "'account_size' (fixed), 'equity' (compounds every trade), or 'daily_equity' (compounds once per day)."
        ),
    )
    risk_per_trade_pct: float = Field(default=1.0, gt=0.0, le=100.0, description="Risk percentage per trade (e.g., 1.0 = 1% of account)")
    account_size: float = Field(default=10000.0, gt=0.0, description="Initial account size for position sizing")


class PortfolioConfig(BaseModel):
    """Portfolio-level risk configuration.

    v1 focus:
    - Enforce a max *open risk* cap (risk-to-stop) by scaling down position size.
    - Provide a last-resort ATR-based stop fallback when stop computation fails.
    """

    enabled: bool = Field(default=False, description="Enable portfolio-level controls")

    # Open-risk (risk-to-stop) cap across all open positions
    max_open_risk_pct: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=100.0,
        description=(
            "Maximum total open risk (risk-to-stop) as % of sizing base. "
            "If set, new entries are scaled down to fit remaining risk budget."
        ),
    )

    # Stop-loss fallback (last resort)
    stop_fallback_enabled: bool = Field(
        default=False,
        description="If True, use ATR-based stop when strategy stop is missing/invalid.",
    )
    stop_fallback_atr_period: int = Field(default=14, gt=0, description="ATR period for fallback stop")
    stop_fallback_atr_multiplier: float = Field(default=3.0, gt=0.0, description="ATR multiplier for fallback stop")


class ExecutionConfig(BaseModel):
    """Execution configuration."""
    max_positions: int = Field(default=1, ge=1)
    fill_timing: Literal["next_open", "same_close"] = Field(
        default="next_open",
        description=(
            "When to fill entries relative to signal evaluation. "
            "'next_open' (default) evaluates on bar close and fills at the next bar open (causal). "
            "'same_close' fills on the same bar close (less realistic; can introduce optimistic bias)."
        ),
    )
    flatten_enabled: bool = Field(
        default=True,
        description="Enable end-of-day flattening of all open positions"
    )
    flatten_time: Optional[str] = Field(
        default="21:30",
        description="Time to flatten all positions (format: 'HH:MM' in UTC+00). Example: '21:30' for 9:30 PM GMT"
    )
    stop_signal_search_minutes_before: Optional[int] = Field(
        default=None,
        ge=0,
        description="Stop searching for new signals X minutes before flatten time. "
                   "Example: If flatten_time is 21:30 and this is 30, signal search stops at 21:00. "
                   "None = no early stop (default)"
    )
    max_wait_bars: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum bars to wait for entry condition after signal. If entry condition not met within this time, signal is skipped. Default: None (no limit)"
    )
    enforce_margin_checks: bool = Field(
        default=True,
        description="Enforce margin requirements before entering positions. Set to False to match NinjaTrader's default backtesting behavior (no margin enforcement). For benchmarking only - keep True for realistic trading."
    )


class TradeLimitConfig(BaseModel):
    """Trade/day and correlation limits."""
    max_trades_per_day: int = Field(default=1, ge=1)
    max_daily_loss_pct: Optional[float] = Field(
        default=3.0,
        gt=0.0,
        le=100.0,
        description=(
            "Stop taking NEW entries for the remainder of the day once realized P&L after costs "
            "is <= -X% of the day base (account_size for fixed sizing, otherwise day-start equity). "
            "Set to None to disable."
        ),
    )
    enable_correlation: bool = True


class TradeDirectionConfig(BaseModel):
    """Enable/disable long/short trades."""
    allow_long: bool = True
    allow_short: bool = True


class ADXFilterConfig(BaseModel):
    """ADX regime filter."""
    enabled: bool = True
    period: int = Field(default=14, gt=0)
    min_forex: float = Field(default=23.0, ge=0.0)
    min_indices: float = Field(default=20.0, ge=0.0)
    min_metals: float = Field(default=25.0, ge=0.0)
    min_commodities: float = Field(default=26.0, ge=0.0)
    min_crypto: float = Field(default=28.0, ge=0.0)


class ATRPercentileFilterConfig(BaseModel):
    """ATR percentile filter."""
    enabled: bool = True
    lookback: int = Field(default=100, gt=0)
    min_percentile: float = Field(default=30.0, ge=0.0, le=100.0)


class EMAExpansionFilterConfig(BaseModel):
    """EMA expansion filter."""
    enabled: bool = True
    lookback: int = Field(default=5, gt=0)


class SwingFilterConfig(BaseModel):
    """Swing structure filter."""
    enabled: bool = False
    lookback: int = Field(default=50, gt=0)
    min_trend_score: float = Field(default=0.60, ge=0.0, le=1.0)


class CompositeRegimeConfig(BaseModel):
    """Composite regime score weighting."""
    enabled: bool = True
    min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    use_adx: bool = True
    use_atr_percentile: bool = True
    use_ema_expansion: bool = True
    use_swing: bool = False
    weight_adx: float = Field(default=1.0, ge=0.0)
    weight_atr_percentile: float = Field(default=1.0, ge=0.0)
    weight_ema_expansion: float = Field(default=1.0, ge=0.0)
    weight_swing: float = Field(default=1.0, ge=0.0)


class TrendQualityConfig(BaseModel):
    """Trend quality filters."""
    enable_ema_distance: bool = False
    min_ema_distance_pips: float = Field(default=5.0, ge=0.0)
    enable_sma_slope: bool = False
    sma_slope_length: int = Field(default=50, gt=0)
    min_sma_slope: float = Field(default=0.1)


class SMADistanceFilterConfig(BaseModel):
    """SMA Distance Quality filter configuration."""
    enabled: bool = False
    min_distance_atr: float = Field(default=0.5, ge=0.0)
    middle_tf: str = Field(default="5m")
    higher_tf: str = Field(default="15m")
    sma_period: int = Field(default=50, gt=0)


class SMASlopeFilterConfig(BaseModel):
    """SMA Slope filter configuration."""
    enabled: bool = False
    min_slope: float = Field(default=0.1)
    middle_tf: str = Field(default="5m")
    higher_tf: str = Field(default="15m")
    middle_lookback: int = Field(default=3, gt=0)
    higher_lookback: int = Field(default=2, gt=0)
    sma_period: int = Field(default=50, gt=0)


class VolumeIncreasingFilterConfig(BaseModel):
    """Volume Increasing filter configuration."""
    enabled: bool = False
    mode: Literal["raw_volume", "volume_oscillator"] = Field(default="volume_oscillator")
    middle_tf: str = Field(default="5m")
    higher_tf: str = Field(default="15m")
    oscillator_fast: int = Field(default=15, gt=0)
    oscillator_slow: int = Field(default=30, gt=0)


class RegimeFiltersConfig(BaseModel):
    """Bundle of regime and trend filters."""
    adx: ADXFilterConfig = Field(default_factory=ADXFilterConfig)
    atr_percentile: ATRPercentileFilterConfig = Field(default_factory=ATRPercentileFilterConfig)
    atr_threshold: Optional[Dict[str, Any]] = None  # ATRThresholdFilterConfig handled separately
    ema_expansion: EMAExpansionFilterConfig = Field(default_factory=EMAExpansionFilterConfig)
    swing: SwingFilterConfig = Field(default_factory=SwingFilterConfig)
    composite: CompositeRegimeConfig = Field(default_factory=CompositeRegimeConfig)
    trend_quality: TrendQualityConfig = Field(default_factory=TrendQualityConfig)
    sma_distance: SMADistanceFilterConfig = Field(default_factory=SMADistanceFilterConfig)
    sma_slope: SMASlopeFilterConfig = Field(default_factory=SMASlopeFilterConfig)


class VolumeFiltersConfig(BaseModel):
    """Volume filters configuration."""
    volume_increasing: VolumeIncreasingFilterConfig = Field(default_factory=VolumeIncreasingFilterConfig)


class BacktestConfig(BaseModel):
    """Backtest configuration."""
    commissions: float = Field(default=0.0004, ge=0.0)
    slippage_ticks: float = Field(default=0.0, ge=0.0)


class StrategyConfig(BaseModel):
    """Complete strategy configuration schema."""
    strategy_name: str
    description: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    market: MarketConfig
    timeframes: TimeframeConfig
    moving_averages: Dict[str, MovingAverageConfig]
    macd: MACDConfig = Field(default_factory=MACDConfig)
    macd_settings: Dict[str, MACDExtendedConfig] = Field(
        default_factory=lambda: {
            "signal": MACDExtendedConfig(),
            "entry": MACDExtendedConfig(min_bars=2),
        }
    )
    ma_settings: Dict[str, MAGroupConfig] = Field(
        default_factory=lambda: {
            "signal": MAGroupConfig(),
            "entry": MAGroupConfig(),
        }
    )
    alignment_rules: Dict[str, AlignmentRulesConfig]
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    partial_exit: PartialExitConfig = Field(default_factory=PartialExitConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    calendar_filters: CalendarFiltersConfig = Field(default_factory=CalendarFiltersConfig)
    news_filter: NewsFilterConfig = Field(default_factory=NewsFilterConfig)
    regime_filters: RegimeFiltersConfig = Field(default_factory=RegimeFiltersConfig)
    volume_filters: VolumeFiltersConfig = Field(default_factory=VolumeFiltersConfig)
    trade_limits: TradeLimitConfig = Field(default_factory=TradeLimitConfig)
    trade_direction: TradeDirectionConfig = Field(default_factory=TradeDirectionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    @field_validator('moving_averages')
    @classmethod
    def validate_moving_averages(cls, v: Dict[str, Any]) -> Dict[str, MovingAverageConfig]:
        """Convert dict to MovingAverageConfig objects."""
        return {k: MovingAverageConfig(**val) if isinstance(val, dict) else val for k, val in v.items()}

    @field_validator('alignment_rules')
    @classmethod
    def validate_alignment_rules(cls, v: Dict[str, Any]) -> Dict[str, AlignmentRulesConfig]:
        """Convert dict to AlignmentRulesConfig objects."""
        return {k: AlignmentRulesConfig(**val) if isinstance(val, dict) else val for k, val in v.items()}

    @field_validator('macd_settings')
    @classmethod
    def validate_macd_settings(cls, v: Dict[str, Any]) -> Dict[str, MACDExtendedConfig]:
        """Normalize MACD settings dict."""
        return {k: MACDExtendedConfig(**val) if isinstance(val, dict) else val for k, val in v.items()}

    @field_validator('ma_settings')
    @classmethod
    def validate_ma_settings(cls, v: Dict[str, Any]) -> Dict[str, MAGroupConfig]:
        """Normalize MA settings dict."""
        return {k: MAGroupConfig(**val) if isinstance(val, dict) else val for k, val in v.items()}


class WalkForwardConfig(BaseModel):
    """Walk-forward analysis configuration."""
    start_date: str
    end_date: str
    window_type: Literal["expanding", "rolling"] = "expanding"
    training_period: Dict[str, Any]
    test_period: Dict[str, Any]
    min_training_period: str = "1 year"
    holdout_end_date: Optional[str] = Field(
        default=None,
        description="End date of holdout period to exclude from walk-forward (e.g., '2025-01-01' excludes 2025+). "
                   "This ensures the most recent year is reserved for one-time final OOS test."
    )
    wf_min_trades_per_period: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum number of trades per test period for statistical significance (default: 30). "
                   "Periods with fewer trades are excluded from statistical calculations but still recorded. "
                   "If None, uses default from OOSValidationCriteria."
    )


class TrainingValidationCriteria(BaseModel):
    """Pass/fail criteria for Phase 1: Training Validation.
    
    These criteria ensure the strategy has a real edge before using OOS data.
    Based on industry standards and Timothy Masters' methodology.
    """
    # Monte Carlo Permutation criteria (legacy - kept for backward compatibility)
    monte_carlo_p_value_max: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum p-value for Monte Carlo test (default 0.05 = 95% confidence). Strategy must beat random."
    )
    monte_carlo_min_percentile: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Minimum percentile rank for observed metric (default 90th percentile)."
    )
    
    # New Monte Carlo Suite criteria
    min_mc_score: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum combined Monte Carlo robustness score (default 0.60 = 60%). Weighted average of all MC tests."
    )
    min_mc_percentile: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Minimum combined Monte Carlo percentile (default 70th percentile). Strategy must outperform random in 70%+ of tests."
    )
    
    # Parameter Sensitivity criteria
    sensitivity_max_cv: float = Field(
        default=0.3,
        ge=0.0,
        description="Maximum coefficient of variation for parameter sensitivity (default 0.3 = 30% variation allowed). Lower is more robust."
    )
    sensitivity_min_trades: int = Field(
        default=30,
        ge=1,
        description="Minimum number of trades required for sensitivity analysis."
    )
    
    # Backtest quality criteria (on training data)
    training_min_profit_factor: float = Field(
        default=1.5,
        gt=1.0,
        description="Minimum profit factor on training data (default 1.5)."
    )
    training_min_sharpe: float = Field(
        default=1.0,
        description="Minimum Sharpe ratio on training data (default 1.0)."
    )
    training_min_trades: int = Field(
        default=30,
        ge=1,
        description="Minimum number of trades required for validation."
    )

    training_max_drawdown_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Maximum allowed max drawdown percentage on training data (default 100 = disabled)."
    )


class OOSValidationCriteria(BaseModel):
    """Pass/fail criteria for Phase 2: Out-of-Sample Validation.
    
    These criteria test the strategy on unseen data (only used once).
    """
    # Walk-Forward Analysis criteria
    wf_min_test_pf: float = Field(
        default=1.5,
        gt=1.0,
        description="Minimum profit factor on test periods (default 1.5)."
    )
    wf_min_consistency_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum consistency score across walk-forward steps (default 0.7 = 70%)."
    )
    wf_max_pf_std: float = Field(
        default=0.5,
        ge=0.0,
        description="Maximum standard deviation of test PF across steps (default 0.5). Lower is more consistent."
    )
    wf_min_test_periods: int = Field(
        default=3,
        ge=1,
        description="Minimum number of test periods required (default 3)."
    )
    wf_min_trades_per_period: int = Field(
        default=30,
        ge=1,
        description="Minimum number of trades per test period for statistical significance (default 30). "
                   "Periods with fewer trades are excluded from statistical calculations but still recorded."
    )
    wf_train_test_pf_ratio_min: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of test PF to train PF (default 0.8 = 80%). Prevents overfitting."
    )
    
    # Walk-Forward Efficiency (WFE) criteria
    wf_min_wfe: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum Walk-Forward Efficiency (default 0.6 = 60%). WFE = mean(test_pf / train_pf) across periods."
    )
    
    # OOS Consistency Score criteria
    wf_min_oos_consistency: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum OOS consistency score (default 0.6 = 60%). Percentage of OOS periods with PF >= threshold."
    )
    
    # Statistical significance criteria
    wf_min_binomial_p_value: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Maximum p-value for binomial test (default 0.10 = 90% confidence). "
                   "Tests if OOS PF >= threshold is statistically significant."
    )
    
    # Overall OOS criteria
    oos_min_sharpe: float = Field(
        default=1.0,
        description="Minimum Sharpe ratio on OOS data (default 1.0)."
    )
    oos_max_drawdown_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable drawdown percentage on OOS data (default 50%)."
    )


class StationarityCriteria(BaseModel):
    """Pass/fail criteria for Phase 3: Stationarity Analysis.
    
    Determines when to retrain the strategy after going live.
    Uses dual-threshold logic: absolute (PF < min_acceptable_pf) + relative (85% degradation).
    Configurable windows for different strategy types (sparse vs frequent).
    """
    degradation_threshold_pct: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Relative degradation threshold (default 0.85 = 85% of baseline)."
    )
    min_acceptable_pf: float = Field(
        default=1.5,
        ge=0.0,
        description="Absolute minimum PF threshold (default 1.5)."
    )
    min_analysis_periods: int = Field(
        default=5,
        ge=1,
        description="Minimum number of analysis periods required (default 5)."
    )
    
    # Walk-Forward Window Configuration
    max_days: int = Field(
        default=30,
        ge=1,
        description="Maximum window size to test in days (default: 30)."
    )
    step_days: int = Field(
        default=1,
        ge=1,
        description="Step size for window testing in days (default: 1)."
    )
    
    # Baseline Windows (3-window approach)
    baseline_window1_start: int = Field(
        default=15,
        ge=1,
        description="First baseline window start in days (default: 15)."
    )
    baseline_window1_end: int = Field(
        default=20,
        ge=1,
        description="First baseline window end in days (default: 20)."
    )
    baseline_window2_start: int = Field(
        default=20,
        ge=1,
        description="Second baseline window start in days (default: 20)."
    )
    baseline_window2_end: int = Field(
        default=25,
        ge=1,
        description="Second baseline window end in days (default: 25)."
    )
    baseline_window3_start: int = Field(
        default=25,
        ge=1,
        description="Third baseline window start in days (default: 25)."
    )
    baseline_window3_end: int = Field(
        default=30,
        ge=1,
        description="Third baseline window end in days (default: 30)."
    )
    
    # Conditional Metrics Thresholds
    win_rate_degradation_pct: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Win rate degradation threshold (default: 0.20 = 20% drop)."
    )
    payoff_degradation_pct: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Payoff ratio degradation threshold (default: 0.30 = 30% drop)."
    )
    big_win_freq_degradation_pct: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Big win frequency degradation threshold (default: 0.50 = 50% drop)."
    )
    stability_cv_threshold: float = Field(
        default=0.20,
        ge=0.0,
        description="Coefficient of variation threshold for stability (default: 0.20 = 20%)."
    )
    
    # Big Win Detection
    big_win_threshold_r: float = Field(
        default=2.0,
        ge=0.0,
        description="R-multiple threshold for 'big win' detection (default: 2.0)."
    )


class ValidationCriteriaConfig(BaseModel):
    """Complete validation criteria configuration.
    
    Users can adjust these thresholds in config/validation_criteria.yml
    Defaults follow industry standards for robust strategy validation.
    """
    training: TrainingValidationCriteria = Field(default_factory=TrainingValidationCriteria)
    oos: OOSValidationCriteria = Field(default_factory=OOSValidationCriteria)
    stationarity: StationarityCriteria = Field(default_factory=StationarityCriteria)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_strategy_config(config_dict: Dict[str, Any]) -> StrategyConfig:
    """Validate and return StrategyConfig object."""
    return StrategyConfig(**config_dict)


def load_and_validate_strategy_config(config_path: Path) -> StrategyConfig:
    """Load and validate strategy configuration from file."""
    config_dict = load_config(config_path)
    return validate_strategy_config(config_dict)


def load_defaults() -> Dict[str, Any]:
    """Load default configuration values."""
    defaults_path = Path(__file__).parent / "defaults.yml"
    return load_config(defaults_path)


def load_validation_criteria(criteria_path: Optional[Path] = None) -> ValidationCriteriaConfig:
    """Load validation criteria configuration.
    
    Args:
        criteria_path: Path to validation criteria YAML file.
                      If None, uses default config/validation_criteria.yml
    
    Returns:
        ValidationCriteriaConfig object
    """
    if criteria_path is None:
        criteria_path = Path(__file__).parent / "validation_criteria.yml"
    
    if not criteria_path.exists():
        # Return defaults if file doesn't exist
        return ValidationCriteriaConfig()
    
    criteria_dict = load_config(criteria_path)
    return ValidationCriteriaConfig(**criteria_dict)


