"""Professional-grade backtesting engine for institutional-style trading systems.

This module implements a complete backtesting engine that:
- Can backtest ANY instrument (FX, futures, crypto, equities, CFDs)
- Is strategy-agnostic
- Correctly handles leverage, margin, contract specs, and P&L by market
- Produces deterministic, reproducible results
- Matches real-world broker behavior as closely as possible
- Is suitable for research, validation, and future live trading integration

Key architectural principles:
1. Strategies NEVER know about leverage, margin, contract size, commission math
2. Strategy outputs ONLY intent: direction, entry_price, stop_price, optional target, risk_pct
3. Engine owns execution, sequencing, and state transitions
4. Broker/Account layer handles all market-specific rules
5. NO market-specific logic in strategies
6. NO leverage logic in strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime

from strategies.base import StrategyBase
from engine.market import MarketSpec
from engine.broker import BrokerModel
from engine.account import AccountState
from engine.resampler import resample_multiple
from engine.trade_management import TradeManagementManager, ExitResolver, ExitCondition, ExitType
try:
    from adapters.data.data_loader import DataLoader
except ImportError:
    # Fallback if data_loader not available
    from adapters.data.csv_loader import CSVDataLoader as DataLoader


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class PartialExit:
    """Partial exit record.
    
    Attributes:
        exit_time: Timestamp when partial exit occurred
        exit_price: Price at which partial exit executed
        quantity: Quantity closed in this partial exit
        pnl: P&L from this partial exit (after costs)
    """
    exit_time: pd.Timestamp
    exit_price: float
    quantity: float
    pnl: float


@dataclass
class RandomizedEntryContext:
    """Context for randomized entry Monte Carlo testing.
    
    This context is used to force entry execution when randomized entry is enabled,
    bypassing normal signal validation while keeping all other logic (stops, exits, risk) identical.
    
    Attributes:
        enabled: Whether randomized entry is active
        entry_bar_index: Bar index at which to force entry (None if not set)
        entry_price: Price at which to force entry (None if not set)
        direction: Direction of the forced entry ('long' or 'short')
    """
    enabled: bool = False
    entry_bar_index: Optional[int] = None
    entry_price: Optional[float] = None
    direction: Optional[str] = None  # 'long' or 'short'


@dataclass
class Position:
    """Open position model.
    
    Attributes:
        instrument: Trading symbol
        direction: 'long' or 'short'
        size: Position size (quantity) - current size after partials
        entry_price: Entry price
        stop_price: Stop loss price (updated by trailing stop)
        target_price: Optional take profit price
        open_time: Entry timestamp
        entry_commission: Commission paid on entry
        margin_used: Margin locked for this position
        unrealized_pnl: Current unrealized P&L (updated each bar)
        initial_size: Original position size (before partials)
        partial_taken: Whether partial exit was executed
        partial_price: Price level for partial exit
        partial_exits: List of partial exit records for this position
        trailing_active: Whether trailing stop is active
        trailing_start_price: Price at which trailing starts
        last_trail_time: Last time trailing stop was updated (for cooldown)
        last_notified_r: Last R multiple notified (for stepped trailing)
        r_value: The R value (stop loss distance) for this trade
    """
    instrument: str
    direction: str
    size: float
    entry_price: float
    stop_price: float
    target_price: Optional[float] = None
    open_time: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    entry_commission: float = 0.0
    margin_used: float = 0.0
    unrealized_pnl: float = 0.0
    initial_size: float = 0.0  # Original size before partials
    partial_taken: bool = False
    partial_price: Optional[float] = None  # Legacy: first partial price
    partial_levels: List[Dict] = field(default_factory=list)  # NEW: Multiple partial levels with prices and exit_pct
    partial_exits: List[PartialExit] = field(default_factory=list)  # Track partial exits
    trailing_active: bool = False
    trailing_start_price: Optional[float] = None
    last_trail_time: Optional[pd.Timestamp] = None
    last_notified_r: float = 0.0
    r_value: float = 0.0  # Stop loss distance in price terms
    tp_levels: List[Dict] = field(default_factory=list)  # NEW: Multiple TP levels with prices and exit_pct
    # MFE/MAE tracking (Maximum Favorable/Adverse Excursion)
    max_favorable_price: Optional[float] = None  # Best price reached (highest for long, lowest for short)
    max_adverse_price: Optional[float] = None  # Worst price reached (lowest for long, highest for short)
    max_favorable_pnl: float = 0.0  # Maximum favorable P&L reached
    max_adverse_pnl: float = 0.0  # Maximum adverse P&L reached
    mfe_reached_at: Optional[pd.Timestamp] = None  # Timestamp when MFE was reached
    mae_reached_at: Optional[pd.Timestamp] = None  # Timestamp when MAE was reached


@dataclass
class Trade:
    """Completed trade model (immutable).
    
    Monte Carlo compatible trade structure with explicit P&L fields.
    
    Attributes:
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        direction: 'long' or 'short'
        entry_price: Entry price
        exit_price: Exit price
        quantity: Position size (REQUIRED, float)
        pnl_raw: Raw P&L before costs
        pnl_after_costs: P&L after all costs (REQUIRED for MC returns)
        commission: Total commission paid
        slippage: Total slippage paid
        stop_price: Stop loss price used
        exit_reason: Reason for exit
        partial_exits: List of partial exit records
    """
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float  # REQUIRED - position size
    pnl_raw: float = 0.0  # Raw P&L before costs
    pnl_after_costs: float = 0.0  # REQUIRED - P&L after all costs (for MC returns)
    commission: float = 0.0
    slippage: float = 0.0
    stop_price: Optional[float] = None
    exit_reason: Optional[str] = None
    partial_exits: List = field(default_factory=list)
    # MFE/MAE (Maximum Favorable/Adverse Excursion)
    mfe_price: Optional[float] = None  # Maximum favorable price reached
    mae_price: Optional[float] = None  # Maximum adverse price reached (industry standard: MAE)
    mfe_pnl: float = 0.0  # Maximum favorable P&L reached
    mae_pnl: float = 0.0  # Maximum adverse P&L reached (industry standard: MAE)
    mfe_r: Optional[float] = None  # MFE in R-multiples
    mae_r: Optional[float] = None  # MAE in R-multiples (industry standard: MAE)
    mfe_reached_at: Optional[pd.Timestamp] = None  # Timestamp when MFE was reached
    mae_reached_at: Optional[pd.Timestamp] = None  # Timestamp when MAE was reached
    
    # Compatibility aliases (for backward compatibility)
    @property
    def size(self) -> float:
        """Alias for quantity (backward compatibility)."""
        return self.quantity
    
    @property
    def gross_pnl(self) -> float:
        """Alias for pnl_raw (backward compatibility)."""
        return self.pnl_raw
    
    @property
    def net_pnl(self) -> float:
        """Alias for pnl_after_costs (backward compatibility)."""
        return self.pnl_after_costs
    
    @property
    def r_multiple(self) -> float:
        """Calculate R multiple (for backward compatibility)."""
        # If an explicit override was provided at construction, use it
        if hasattr(self, '_r_multiple_override') and self._r_multiple_override is not None:
            return float(self._r_multiple_override)

        if self.stop_price is None:
            return 0.0
        initial_risk = abs(self.entry_price - self.stop_price) * self.quantity
        if initial_risk > 0:
            return self.pnl_after_costs / initial_risk
        return 0.0

    def __init__(
        self,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: Optional[float] = None,
        pnl_raw: Optional[float] = None,
        pnl_after_costs: Optional[float] = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        stop_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
        partial_exits: Optional[List] = None,
        # Legacy / alternate kwargs
        size: Optional[float] = None,
        gross_pnl: Optional[float] = None,
        net_pnl: Optional[float] = None,
        mfe_price: Optional[float] = None,
        mae_price: Optional[float] = None,
        mfe_pnl: float = 0.0,
        mae_pnl: float = 0.0,
        mfe_r: Optional[float] = None,
        mae_r: Optional[float] = None,
        mfe_reached_at: Optional[pd.Timestamp] = None,
        mae_reached_at: Optional[pd.Timestamp] = None,
        r_multiple: Optional[float] = None,
    ):
        # Map legacy names to canonical fields
        qty = quantity if quantity is not None else (size if size is not None else 0.0)
        raw = pnl_raw if pnl_raw is not None else (gross_pnl if gross_pnl is not None else 0.0)
        after = pnl_after_costs if pnl_after_costs is not None else (net_pnl if net_pnl is not None else 0.0)

        # Assign core fields
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.quantity = float(qty)
        self.pnl_raw = float(raw)
        self.pnl_after_costs = float(after)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.stop_price = stop_price
        self.exit_reason = exit_reason
        self.partial_exits = partial_exits or []

        # r_multiple override (legacy tests may pass this)
        self._r_multiple_override = float(r_multiple) if r_multiple is not None else None

        # MFE/MAE
        self.mfe_price = mfe_price
        self.mae_price = mae_price
        self.mfe_pnl = mfe_pnl
        self.mae_pnl = mae_pnl
        self.mfe_r = mfe_r
        self.mae_r = mae_r
        self.mfe_reached_at = mfe_reached_at
        self.mae_reached_at = mae_reached_at


@dataclass
class BacktestResult:
    """Backtest results container - Monte Carlo compatible.
    
    IMPORTANT: equity_curve uses integer index (0..N) for MC compatibility.
    trade_returns are precomputed: return_i = pnl_after_costs_i / equity_before_trade_i
    
    Attributes:
        initial_capital: Starting capital
        final_capital: Ending capital (equity after all trades)
        equity_curve: Equity curve with integer index (length N+1, where N = trades)
            equity_curve[0] = initial_capital
            equity_curve[i] = equity after trade i
        trade_returns: Array of trade returns (length N, matches trades)
            return_i = pnl_after_costs_i / equity_before_trade_i
        trades: List of completed Trade objects
        total_pnl: Total P&L (final_capital - initial_capital)
        total_trades: Number of completed trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        total_commission: Total commission paid
        total_slippage: Total slippage paid
        max_drawdown: Maximum drawdown percentage (computed from equity_curve)
        strategy_name: Strategy name
        symbol: Trading symbol
    """
    initial_capital: float = 0.0
    final_capital: float = 0.0
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series([], dtype=float))
    trade_returns: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    max_drawdown: float = 0.0
    strategy_name: str = ""
    symbol: str = ""
    
    # Optional fields for backward compatibility
    win_rate: float = 0.0
    drawdown_curve: Optional[pd.Series] = None
    exposure_stats: Dict = field(default_factory=dict)
    leverage_stats: Dict = field(default_factory=dict)
    margin_utilization: Optional[pd.Series] = None

    # Optional fields for research/debugging
    entry_tf: str = ""
    price_df: Optional[pd.DataFrame] = None


# ============================================================================
# Backtest Engine
# ============================================================================

class BacktestEngine:
    """Professional-grade backtesting engine.
    
    This engine implements industry-standard backtesting with:
    - Bar-by-bar execution (no lookahead bias)
    - Proper margin and leverage handling
    - Realistic commission and slippage
    - Support for trailing stops and partial exits
    - Margin call detection and forced liquidation
    - Complete capital tracking and validation
    
    The engine is completely strategy-agnostic and market-agnostic.
    All market-specific logic is handled by MarketSpec and BrokerModel.
    """
    
    def __init__(
        self,
        strategy: StrategyBase,
        market_spec: Optional[MarketSpec] = None,
        initial_capital: float = 10000.0,
        commission_rate: Optional[float] = None,
        slippage_ticks: Optional[float] = None,
        enforce_margin_checks: bool = True,
        randomized_entry_context: Optional[RandomizedEntryContext] = None
    ):
        """
        Initialize backtest engine.
        
        Args:
            strategy: Strategy instance (must implement StrategyBase)
            market_spec: Market specification (auto-loaded if None)
            initial_capital: Starting capital
            commission_rate: Override commission rate (uses market_spec if None)
            slippage_ticks: Override slippage (uses market_spec if None)
            enforce_margin_checks: Whether to enforce margin requirements (True by default for realistic backtesting)
            randomized_entry_context: Context for randomized entry MC testing (None for normal backtests)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.enforce_margin_checks = enforce_margin_checks  # Store as instance variable
        self.randomized_entry_context = randomized_entry_context or RandomizedEntryContext(enabled=False)
        
        # Initialize trade management manager (merges master + strategy configs)
        self.trade_mgmt = TradeManagementManager.from_strategy(strategy.config)
        
        # Initialize exit resolver for handling simultaneous exits
        self.exit_resolver = ExitResolver()
        
        # Initialize logger with file and console handlers
        self.logger = logging.getLogger(__name__)

        # Optional: expose entry-timeframe dataframe on BacktestResult for reporting/tests.
        self._result_entry_tf: Optional[str] = None
        self._result_price_df: Optional[pd.DataFrame] = None
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            logs_dir = Path('data/logs')
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = logs_dir / f'backtest_{timestamp}.log'
            
            # File handler - saves all logs to file
            file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Console handler - shows warnings and errors on console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)  # Only show warnings and errors on console
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # Set logger level to DEBUG (file gets everything, console gets WARNING+)
            self.logger.setLevel(logging.DEBUG)
            
            # Store log file path for reference
            self.log_file_path = log_filename
            self.logger.info(f"Logging initialized. Log file: {log_filename}")
        exec_cfg = getattr(strategy.config, "execution", None)
        self.flatten_enabled = getattr(exec_cfg, "flatten_enabled", True)
        self.flatten_time_str = getattr(exec_cfg, "flatten_time", "21:30") if self.flatten_enabled else None
        self.stop_signal_search_minutes_before = getattr(exec_cfg, "stop_signal_search_minutes_before", None)
        self.fill_timing = getattr(exec_cfg, "fill_timing", "next_open")
        self._flatten_minutes = None
        self._stop_signal_search_minutes = None
        self._flatten_closed_date: Optional[pd.Timestamp] = None
        
        # Load market spec if not provided
        if market_spec is None:
            symbol = getattr(strategy.config.market, 'symbol', 'EURUSD')
            try:
                market_spec = MarketSpec.load_from_profiles(symbol)
            except ValueError:
                # Fallback: create from strategy config
                market_config = strategy.config.market
                asset_class = 'forex' if 'USD' in symbol and len(symbol) == 6 else 'crypto'
                
                # Set forex defaults if asset class is forex
                contract_size = None
                pip_value = None
                if asset_class == 'forex':
                    contract_size = 100000  # Standard lot size
                    pip_value = 0.0001  # Standard pip size for most pairs
                
                market_spec = MarketSpec(
                    symbol=getattr(market_config, 'symbol', symbol),
                    exchange=getattr(market_config, 'exchange', 'unknown'),
                    asset_class=asset_class,
                    market_type=getattr(market_config, 'market_type', 'spot'),
                    leverage=getattr(market_config, 'leverage', 1.0),
                    contract_size=contract_size,
                    pip_value=pip_value,
                    commission_rate=getattr(strategy.config.backtest, 'commissions', 0.0004),
                    slippage_ticks=getattr(strategy.config.backtest, 'slippage_ticks', 0.0)
                )
        
        self.market_spec = market_spec
        
        # Override commission/slippage if provided
        if commission_rate is not None:
            self.market_spec.commission_rate = commission_rate
            # For futures, also override commission_per_contract if commission_rate is 0
            # (This allows disabling commission for futures by setting commission_rate=0)
            if market_spec.asset_class == 'futures' and commission_rate == 0.0:
                self.market_spec.commission_per_contract = 0.0
                self.logger.info(f"Disabled futures commission: commission_per_contract set to 0.0")
        if slippage_ticks is not None:
            self.market_spec.slippage_ticks = slippage_ticks
            if slippage_ticks == 0.0:
                self.logger.info(f"Disabled slippage: slippage_ticks set to 0.0")
        
        # Initialize broker
        self.broker = BrokerModel(self.market_spec)
        
        # Initialize account
        self.account = AccountState(cash=initial_capital)
        
        # Track trades and positions
        self.trades: List[Trade] = []
        self.positions: List[Position] = []
        
        # Track equity curve (bar-by-bar for reporting)
        # CRITICAL: equity_curve is derived from equity_list + account.equity updates
        # equity_list is the single source of truth for trade-based equity
        self.equity_curve: List[float] = []
        self.equity_timestamps: List[pd.Timestamp] = []
        
        # Track margin utilization
        self.margin_utilization: List[float] = []
        
        # Data loader
        self.data_loader = DataLoader()
        
        # Parse flatten time (HH:MM) if provided
        if self.flatten_enabled and self.flatten_time_str:
            try:
                hours, mins = self.flatten_time_str.split(":")
                self._flatten_minutes = int(hours) * 60 + int(mins)
                
                # Calculate stop signal search time if configured
                if self.stop_signal_search_minutes_before is not None:
                    self._stop_signal_search_minutes = self._flatten_minutes - self.stop_signal_search_minutes_before
                    # Handle wrap-around (if stop time is negative, it means previous day)
                    if self._stop_signal_search_minutes < 0:
                        self._stop_signal_search_minutes += 24 * 60
            except Exception:
                self._flatten_minutes = None
                self._stop_signal_search_minutes = None
        
        # Load trade limits config
        trade_limits_cfg = getattr(strategy.config, 'trade_limits', None)
        self.max_trades_per_day = getattr(trade_limits_cfg, 'max_trades_per_day', None) if trade_limits_cfg else None
        self.max_daily_loss_pct = getattr(trade_limits_cfg, 'max_daily_loss_pct', None) if trade_limits_cfg else None

        # Track entries per day: {date: count}
        # (Used to enforce max_trades_per_day. This counts successful ENTRIES, not exits.)
        self._trades_per_day: Dict[pd.Timestamp, int] = {}

        # Track realized P&L after costs per day (includes partial exits)
        self._realized_pnl_after_costs_per_day: Dict[pd.Timestamp, float] = {}

        # Track day-start equity for daily sizing / daily loss baselines
        self._day_start_equity_by_day: Dict[pd.Timestamp, float] = {}
        self._current_day_key: Optional[pd.Timestamp] = None

        # Debug counters
        self._daily_loss_rejections: int = 0

    @staticmethod
    def _day_key(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.normalize()

    def _ensure_day_initialized(self, ts: pd.Timestamp) -> None:
        day_key = self._day_key(ts)
        self._current_day_key = day_key
        if day_key not in self._day_start_equity_by_day:
            # Snapshot equity at first bar of the day (may include any carried positions)
            self._day_start_equity_by_day[day_key] = float(self.account.equity)

    def _get_day_base_equity_for_limits(self, ts: pd.Timestamp) -> float:
        """Baseline used for daily loss % checks."""
        risk_cfg = getattr(self.strategy.config, 'risk', None)
        sizing_mode = getattr(risk_cfg, 'sizing_mode', 'account_size') if risk_cfg else 'account_size'
        if sizing_mode == 'account_size':
            acct_size = getattr(risk_cfg, 'account_size', None) if risk_cfg else None
            return float(acct_size) if acct_size is not None else float(self.initial_capital)

        day_key = self._day_key(ts)
        base = self._day_start_equity_by_day.get(day_key)
        if base is None:
            base = float(self.account.equity)
            self._day_start_equity_by_day[day_key] = base
        return float(base)

    def _daily_loss_limit_reached(self, ts: pd.Timestamp) -> bool:
        if self.max_daily_loss_pct is None:
            return False
        try:
            max_daily_loss_pct = float(self.max_daily_loss_pct)
        except (TypeError, ValueError):
            return False
        if max_daily_loss_pct <= 0:
            return False

        day_key = self._day_key(ts)
        realized = float(self._realized_pnl_after_costs_per_day.get(day_key, 0.0))
        base = self._get_day_base_equity_for_limits(ts)
        limit_amount = base * (max_daily_loss_pct / 100.0)
        return realized <= -limit_amount

    def _get_entry_fill(
        self,
        df_entry: pd.DataFrame,
        decision_bar_idx: int,
        decision_time: pd.Timestamp,
        decision_close: float,
    ) -> Optional[tuple[pd.Timestamp, float]]:
        """Compute fill time/price for an entry.

        Contract:
        - Strategies generate signals using information available at/through the decision bar close.
        - Engine applies fill timing for execution. Default is 'next_open' (causal).
        """
        if self.fill_timing == "same_close":
            return decision_time, float(decision_close)

        # Default: next bar open
        next_idx = decision_bar_idx + 1
        if next_idx >= len(df_entry):
            return None
        next_time = df_entry.index[next_idx]
        next_open = float(df_entry.iloc[next_idx]["open"])
        return next_time, next_open
    
    def run(
        self,
        data: Union[Path, pd.DataFrame],
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> BacktestResult:
        """
        Run backtest on provided data.
        
        Args:
            data: Path to data file or DataFrame
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            BacktestResult object with complete backtest results
        """
        # Load and prepare data
        if isinstance(data, Path) or isinstance(data, str):
            df_base = self.data_loader.load(Path(data))
        else:
            df_base = data.copy()
        
        # Filter by date if provided
        if start_date is not None:
            df_base = df_base[df_base.index >= start_date]
        if end_date is not None:
            df_base = df_base[df_base.index < end_date]
        
        if len(df_base) == 0:
            raise ValueError("No data available after filtering")
        
        # Prepare multi-timeframe data
        required_tfs = self.strategy.get_required_timeframes()
        df_by_tf = resample_multiple(df_base, required_tfs)
        df_by_tf['base'] = df_base  # Add base timeframe
        
        # Calculate indicators
        df_by_tf = self.strategy.prepare_data(df_by_tf)
        
        # Apply warm-up period if strategy defines it
        warmup_bars = getattr(self.strategy, 'get_warmup_bars', None)
        if warmup_bars and callable(warmup_bars):
            warmup_count = warmup_bars()
            if warmup_count > 0:
                # Filter out warm-up bars from entry timeframe
                entry_tf = self.strategy.config.timeframes.entry_tf
                if entry_tf in df_by_tf and len(df_by_tf[entry_tf]) > warmup_count:
                    df_by_tf[entry_tf] = df_by_tf[entry_tf].iloc[warmup_count:]
                    self.logger.info(f"Applied warm-up period: skipped first {warmup_count} bars")
        
        # Generate signals
        signals = self.strategy.generate_signals(df_by_tf)
        
        # Get entry timeframe for bar-by-bar execution (needed even if no signals)
        entry_tf = self.strategy.config.timeframes.entry_tf
        df_entry = df_by_tf[entry_tf]

        # Stash for BacktestResult consumers (tests/reports)
        self._result_entry_tf = str(entry_tf)
        self._result_price_df = df_entry
        
        # Initialize Monte Carlo compatible tracking (needed even if no signals)
        # - equity_list: equity after each trade (for integer-indexed equity_curve)
        # - trade_returns_list: return_i = pnl_after_costs_i / equity_before_trade_i
        self.equity_list: List[float] = [self.initial_capital]  # Start with initial capital
        self.trade_returns_list: List[float] = []  # Will be computed as trades close
        
        if len(signals) == 0:
            warnings.warn("No signals generated by strategy")
            return self._create_result(df_entry.index[0], df_entry.index[-1])
        
        # Log signal count
        self.logger.debug(f"Generated {len(signals)} signals")
        
        # Pending signals queue (for waiting-entry logic)
        self.pending_signals: List[Dict] = []
        
        # Track cumulative rejection counts (for debugging)
        self._cumulative_rejections = {'expired': 0, 'no_entry_bars': 0, 'ema_alignment': 0, 
                                      'macd_progression': 0, 'ema_gating': 0, 'invalid_stop': 0, 
                                      'zero_quantity': 0, 'cannot_afford': 0}
        
        # Initialize account equity curve
        self.account.record_equity(df_entry.index[0])
        self.equity_curve.append(self.account.equity)
        self.equity_timestamps.append(df_entry.index[0])
        # Initialize margin utilization (0 at start since no positions)
        self.margin_utilization.append(0.0)
        
        # Bar-by-bar execution
        for idx in range(len(df_entry)):
            current_time = df_entry.index[idx]
            bar = df_entry.iloc[idx]
            current_price = float(bar['close'])

            # Track day boundaries for daily sizing / daily loss limits
            self._ensure_day_initialized(current_time)
            
            # Store current bar index for randomized entry context checking
            self._current_bar_index = idx
            
            # 1. Process exits FIRST (before entries)
            self._process_exits(current_time, current_price, bar, df_by_tf)
            
            # 2. Check for margin calls
            self._check_margin_calls(current_time, current_price)
            
            # 3. Optional flatten at configured time (end-of-day)
            self._check_flatten_time(current_time, current_price)
            
            # 4. Process entries (respects stop_signal_search_minutes_before internally)
            self._process_entries(current_time, current_price, signals, df_by_tf)
            
            # 5. Update account state
            self._update_account_state(current_time, current_price)
        
        # Close any remaining positions at end of data
        final_price = float(df_entry.iloc[-1]['close'])
        final_time = df_entry.index[-1]
        self._close_all_positions(final_time, final_price, 'end_of_data')
        
        # Final account update
        self._update_account_state(final_time, final_price)
        
        # Validate accounting
        self._validate_accounting(final_price)
        
        # Create result
        return self._create_result(df_entry.index[0], df_entry.index[-1])
    
    def _check_ema_alignment_during_trade(
        self,
        position: Position,
        current_bar: pd.Series,
        df_by_tf: Dict[str, pd.DataFrame]
    ) -> Optional[int]:
        """Check EMA alignment during open position.
        
        Returns:
            1 for long alignment, -1 for short alignment, 0 for no alignment, None if check not available
        """
        # Get entry timeframe data
        entry_tf = self.strategy.config.timeframes.entry_tf
        if entry_tf not in df_by_tf:
            return None
        
        df_entry = df_by_tf[entry_tf]
        
        # Find current bar in entry timeframe
        current_time = current_bar.name if hasattr(current_bar, 'name') else None
        if current_time is None:
            return None
        
        entry_bars = df_entry[df_entry.index <= current_time]
        if len(entry_bars) == 0:
            return None
        
        entry_row = entry_bars.iloc[-1]
        entry_ma_cfg = self.strategy.config.ma_settings.get('entry', None)
        if entry_ma_cfg is None:
            return None
        
        # Check alignment using strategy's method
        direction_int = 1 if position.direction == 'long' else -1
        is_aligned = self.strategy._check_ema_alignment(entry_row, entry_ma_cfg, direction_int)
        
        if is_aligned:
            return direction_int
        else:
            return 0  # Alignment broken
    
    def _calculate_current_r_multiple(
        self,
        position: Position,
        current_price: float
    ) -> float:
        """Calculate current R multiple for a position.
        
        Args:
            position: Open position
            current_price: Current market price
            
        Returns:
            Current R multiple (e.g., 1.5 means 1.5R profit)
        """
        if position.r_value <= 0:
            return 0.0
        
        # Calculate current P&L
        if position.direction == 'long':
            price_change = current_price - position.entry_price
        else:  # short
            price_change = position.entry_price - current_price
        
        # R multiple = price_change / r_value
        r_multiple = price_change / position.r_value
        return r_multiple
    
    def _calculate_stepped_trailing_sl(
        self,
        position: Position,
        current_r: float
    ) -> Optional[float]:
        """Calculate stepped trailing stop loss based on R multiple.
        
        Args:
            position: Open position
            current_r: Current R multiple
            
        Returns:
            New stop loss price, or None if should use original SL
        """
        if current_r >= 3.0:
            # At 3R: Trail to +2R
            if position.direction == 'long':
                return position.entry_price + (2.0 * position.r_value)
            else:
                return position.entry_price - (2.0 * position.r_value)
        elif current_r >= 2.0:
            # At 2R: Trail to +1R
            if position.direction == 'long':
                return position.entry_price + (1.0 * position.r_value)
            else:
                return position.entry_price - (1.0 * position.r_value)
        elif current_r >= 1.0:
            # At 1R: Trail to +0.5R
            if position.direction == 'long':
                return position.entry_price + (0.5 * position.r_value)
            else:
                return position.entry_price - (0.5 * position.r_value)
        elif current_r >= 0.5:
            # At 0.5R: Trail to breakeven
            return position.entry_price
        else:
            # Below 0.5R: Use original SL
            return None
    
    def _get_trailing_ema_sl(
        self,
        position: Position,
        current_bar: pd.Series,
        df_by_tf: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """Get trailing stop loss from EMA.
        
        Args:
            position: Open position
            current_bar: Current bar data
            df_by_tf: DataFrames by timeframe
            
        Returns:
            EMA-based stop loss price, or None if not available
        """
        trailing_config = getattr(self.strategy.config, 'trailing_stop', None)
        if not trailing_config or trailing_config.type != 'EMA':
            return None
        
        entry_tf = self.strategy.config.timeframes.entry_tf
        if entry_tf not in df_by_tf:
            return None
        
        df_entry = df_by_tf[entry_tf]
        current_time = current_bar.name if hasattr(current_bar, 'name') else None
        if current_time is None:
            return None
        
        entry_bars = df_entry[df_entry.index <= current_time]
        if len(entry_bars) < 2:
            return None
        
        # Get EMA value from previous closed bar (index 1)
        # Strategy calculates 'trail_ema' column for trailing stop
        ema_col = 'trail_ema'
        
        if ema_col not in entry_bars.columns:
            # Recalculate indicators if not present
            df_entry_with_ema = self.strategy.get_indicators(df_entry, tf='entry')
            entry_bars = df_entry_with_ema[df_entry_with_ema.index <= current_time]
            if len(entry_bars) < 2:
                return None
        
        entry_row = entry_bars.iloc[-2]  # Previous closed bar
        
        if ema_col not in entry_row:
            return None
        
        ema_value = entry_row[ema_col]
        if pd.isna(ema_value):
            return None
        
        return float(ema_value)
    
    def _update_trailing_stop(
        self,
        position: Position,
        current_time: pd.Timestamp,
        current_price: float,
        current_bar: pd.Series,
        df_by_tf: Dict[str, pd.DataFrame],
        is_new_bar: bool = True
    ) -> bool:
        """Update trailing stop for a position.
        
        Args:
            position: Open position
            current_time: Current timestamp
            current_price: Current market price
            current_bar: Current bar data
            df_by_tf: DataFrames by timeframe
            is_new_bar: Whether this is a new bar (for cooldown)
            
        Returns:
            True if stop loss was updated, False otherwise
        """
        # Get trailing stop config from trade management manager
        trailing_config_dict = self.trade_mgmt.get_trailing_stop_config()
        if not trailing_config_dict.get('enabled', False):
            return False
        
        # For backward compatibility, also check strategy config directly
        trailing_config = getattr(self.strategy.config, 'trailing_stop', None)
        
        # Check cooldown period (60 seconds minimum between modifications)
        if position.last_trail_time is not None and is_new_bar:
            time_diff = (current_time - position.last_trail_time).total_seconds()
            if time_diff < 60:
                return False  # Cooldown active
        
        # Check if trailing should start
        if not position.trailing_active:
            if position.trailing_start_price is None:
                return False
            
            should_start = False
            if position.direction == 'long':
                if current_price >= position.trailing_start_price:
                    should_start = True
            else:  # short
                if current_price <= position.trailing_start_price:
                    should_start = True
            
            if should_start:
                position.trailing_active = True
            else:
                return False  # Not yet at trailing level
        
        # Calculate new stop loss using new config structure
        new_sl = None
        trailing_type = trailing_config_dict.get('type', 'EMA')
        stepped = trailing_config_dict.get('stepped', True)
        
        if stepped:
            # Stepped R-based trailing
            current_r = self._calculate_current_r_multiple(position, current_price)
            stepped_sl = self._calculate_stepped_trailing_sl(position, current_r)
            
            # Also get MA-based SL if type is EMA or SMA
            ma_sl = None
            if trailing_type == 'EMA':
                ma_sl = self._get_trailing_ema_sl(position, current_bar, df_by_tf)
            elif trailing_type == 'SMA':
                # TODO: Implement SMA-based trailing
                pass
            
            if stepped_sl is not None:
                new_sl = stepped_sl
                if ma_sl is not None:
                    # Choose the better SL (more protective)
                    if position.direction == 'long':
                        new_sl = max(stepped_sl, ma_sl)
                    else:  # short
                        new_sl = min(stepped_sl, ma_sl)
        else:
            # MA-based trailing only
            if trailing_type == 'EMA':
                new_sl = self._get_trailing_ema_sl(position, current_bar, df_by_tf)
            elif trailing_type == 'SMA':
                # TODO: Implement SMA-based trailing
                pass
            # TODO: Support ATR, percentage, fixed_distance trailing types
        
        if new_sl is None:
            return False
        
        # Get instrument-specific minimum move
        min_move_pips = trailing_config_dict.get('min_move_pips', 7.0)
        symbol = position.instrument
        
        # Check overrides
        min_move_overrides = trailing_config_dict.get('min_move_overrides', {})
        if min_move_overrides:
            for key, value in min_move_overrides.items():
                if key in symbol:
                    min_move_pips = value
                    break
        
        pip_size = self.market_spec.pip_value or 0.0001
        min_move_price = min_move_pips * pip_size
        
        # Check if we should move SL
        should_move = False
        current_sl = position.stop_price
        
        if position.direction == 'long':
            # Long: new SL must be higher and at least min_move away
            if new_sl > current_sl and (new_sl - current_sl) >= min_move_price:
                should_move = True
        else:  # short
            # Short: new SL must be lower and at least min_move away
            if new_sl < current_sl and (current_sl - new_sl) >= min_move_price:
                should_move = True
        
        if should_move:
            position.stop_price = new_sl
            position.last_trail_time = current_time
            
            # Update last_notified_r for stepped trailing
            if stepped:
                current_r = self._calculate_current_r_multiple(position, current_price)
                current_r_int = int(current_r)
                last_notified_r_int = int(position.last_notified_r)
                
                if current_r_int > last_notified_r_int and current_r_int >= 1:
                    position.last_notified_r = current_r
                    # Could log notification here if needed
            
            return True
        
        return False
    
    def _check_partial_execution(
        self,
        position: Position,
        current_price: float,
        current_time: pd.Timestamp
    ) -> bool:
        """Check and execute partial exit if conditions are met.
        
        Supports multiple partial levels (new architecture) with backward compatibility.
        
        Args:
            position: Open position
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            True if partial was executed, False otherwise
        """
        partial_config_dict = self.trade_mgmt.get_partial_exit_config()
        if not partial_config_dict.get('enabled', False):
            return False
        
        # Use new multi-level system if available, otherwise fall back to legacy
        if position.partial_levels:
            # NEW: Multiple partial levels
            executed_any = False
            for level in position.partial_levels:
                if level.get('executed', False):
                    continue  # Already executed this level
                
                level_price = level.get('price')
                if level_price is None:
                    continue
                
                # Check if level reached
                level_hit = False
                if position.direction == 'long':
                    if current_price >= level_price:
                        level_hit = True
                else:  # short
                    if current_price <= level_price:
                        level_hit = True
                
                if level_hit:
                    # Execute partial at this level
                    exit_pct = level.get('exit_pct', 50.0) / 100.0
                    partial_size = position.initial_size * exit_pct
                    partial_size = min(partial_size, position.size)
                    
                    if partial_size > 0:
                        self._execute_partial_exit(position, current_price, current_time, partial_size)
                        level['executed'] = True
                        executed_any = True
            
            return executed_any
        else:
            # LEGACY: Single partial exit
            if position.partial_taken or position.partial_price is None:
                return False
            
            # Check if partial level reached
            partial_hit = False
            if position.direction == 'long':
                if current_price >= position.partial_price:
                    partial_hit = True
            else:  # short
                if current_price <= position.partial_price:
                    partial_hit = True
            
            if not partial_hit:
                return False
            
            # Get exit percentage from config
            exit_pct = partial_config_dict.get('exit_pct', 80.0) / 100.0
            partial_size = position.initial_size * exit_pct
            partial_size = min(partial_size, position.size)
            
            if partial_size <= 0:
                return False
            
            self._execute_partial_exit(position, current_price, current_time, partial_size)
            position.partial_taken = True
            return True
    
    def _execute_partial_exit(
        self,
        position: Position,
        current_price: float,
        current_time: pd.Timestamp,
        partial_size: float
    ) -> None:
        """Execute a partial exit (shared by legacy and new multi-level system).
        
        Args:
            position: Open position
            current_price: Current market price
            current_time: Current timestamp
            partial_size: Size to exit
        """
        
        # Execute partial exit
        exit_price_adj = self.broker.apply_slippage(current_price, partial_size, is_entry=False)
        exit_commission = self.broker.calculate_commission(exit_price_adj, partial_size)
        
        # Calculate P&L for partial
        if position.direction == 'long':
            pnl = (exit_price_adj - position.entry_price) * partial_size
        else:  # short
            pnl = (position.entry_price - exit_price_adj) * partial_size
        
        # MONTE CARLO COMPATIBLE: Track equity before partial trade
        # CRITICAL: Use the LAST recorded equity (after previous trade), not current account equity
        if len(self.equity_list) > 0:
            equity_before_partial = self.equity_list[-1]  # Last recorded equity after previous trade
        else:
            equity_before_partial = self.initial_capital  # First trade
        
        # Calculate P&L after costs
        pnl_raw_partial = pnl  # Raw P&L before costs
        pnl_after_costs_partial = pnl - exit_commission
        
        # Update account (commission reduces cash)
        self.account.cash += (pnl - exit_commission)
        self.account.commission_paid += exit_commission
        slippage_price = self.broker._slippage_price_units()
        self.account.slippage_paid += slippage_price * abs(partial_size)
        
        # Update position
        position.size -= partial_size
        position.margin_used -= position.margin_used * (partial_size / position.initial_size)
        position.partial_taken = True
        
        # CRITICAL FIX: Track partial exit in position for later inclusion in Trade record
        partial_exit_record = PartialExit(
            exit_time=current_time,
            exit_price=exit_price_adj,
            quantity=partial_size,
            pnl=pnl_after_costs_partial
        )
        position.partial_exits.append(partial_exit_record)
        
        # MONTE CARLO COMPATIBLE: Calculate trade return BEFORE updating equity
        if equity_before_partial > 0:
            trade_return_partial = pnl_after_costs_partial / equity_before_partial
        else:
            trade_return_partial = 0.0
        
        # MONTE CARLO COMPATIBLE: Calculate equity after partial trade
        # CRITICAL: Use simple addition (equity_before + pnl_after_costs) for proper compounding
        equity_after_partial = equity_before_partial + pnl_after_costs_partial
        
        # Create partial trade record (logged for analytics; not counted toward max trades)
        partial_trade = Trade(
            entry_time=position.open_time,
            exit_time=current_time,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price_adj,
            quantity=partial_size,  # REQUIRED field
            pnl_raw=pnl_raw_partial,  # Raw P&L before costs
            pnl_after_costs=pnl_after_costs_partial,  # REQUIRED for MC returns
            commission=exit_commission,
            slippage=slippage_price * abs(partial_size),
            stop_price=position.stop_price,
            exit_reason='partial',
            partial_exits=[]  # Partial trades don't have nested partials
        )
        self.trades.append(partial_trade)

        # Track realized P&L after costs for daily loss caps
        day_key = current_time.normalize()
        self._realized_pnl_after_costs_per_day[day_key] = (
            self._realized_pnl_after_costs_per_day.get(day_key, 0.0) + float(pnl_after_costs_partial)
        )
        
        # MONTE CARLO COMPATIBLE: Track equity and returns for partial trade
        self.equity_list.append(equity_after_partial)
        self.trade_returns_list.append(trade_return_partial)
        
        return True
    
    def _process_exits(
        self,
        current_time: pd.Timestamp,
        current_price: float,
        bar: pd.Series,
        df_by_tf: Dict[str, pd.DataFrame]
    ) -> None:
        """Process exit conditions for all open positions.
        
        Exit conditions checked in order:
        1. EMA alignment check (close if broken)
        2. Partial execution check
        3. Trailing stop update
        4. Stop loss (may have been updated by trailing)
        5. Take profit
        
        Args:
            current_time: Current bar timestamp
            current_price: Current market price
            bar: Current bar data
            df_by_tf: DataFrames by timeframe
        """
        positions_to_close = []
        
        # Track if we're on a new bar for trailing stop cooldown
        # For simplicity, assume each call is a new bar (can be improved with bar tracking)
        is_new_bar = True
        
        for pos_idx, position in enumerate(self.positions):
            # Get bar high/low for intrabar checks
            bar_low = float(bar['low']) if 'low' in bar else current_price
            bar_high = float(bar['high']) if 'high' in bar else current_price
            
            # 1. Check EMA alignment during trade (matches MQL5 ManageActiveTrades)
            ema_alignment = self._check_ema_alignment_during_trade(position, bar, df_by_tf)
            if ema_alignment is not None:
                direction_int = 1 if position.direction == 'long' else -1
                if ema_alignment != direction_int:
                    positions_to_close.append((pos_idx, 'ema_alignment_broken'))
                    continue
            
            # 2. Check partial execution (matches MQL5 CheckPartialExecution)
            self._check_partial_execution(position, current_price, current_time)
            
            if position.size <= 0:
                positions_to_close.append((pos_idx, 'partial_full'))
                continue
            
            # 3. Update trailing stop (matches MQL5 UpdateTrailingStop)
            self._update_trailing_stop(
                position,
                current_time,
                current_price,
                bar,
                df_by_tf,
                is_new_bar=is_new_bar
            )
            
            # 4. Collect all potential exit conditions for resolution
            exit_conditions = []
            
            # Hard stop (always highest priority) - check intrabar
            stop_hit = False
            if position.direction == 'long':
                if bar_low <= position.stop_price:
                    stop_hit = True
            else:  # short
                if bar_high >= position.stop_price:
                    stop_hit = True
            
            if stop_hit:
                # Calculate P&L at stop price
                if position.direction == 'long':
                    stop_pnl = (position.stop_price - position.entry_price) * position.size
                else:  # short
                    stop_pnl = (position.entry_price - position.stop_price) * position.size
                
                exit_conditions.append(ExitCondition(
                    exit_type=ExitType.HARD_STOP,
                    exit_price=position.stop_price,
                    pnl=stop_pnl,
                    r_multiple=-1.0,  # Stop loss = -1R
                    exit_pct=100.0,
                    priority=0  # Highest priority
                ))
            
            # Trailing stop (if active and better than hard stop)
            if position.trailing_active:
                current_r = self._calculate_current_r_multiple(position, current_price)
                if position.direction == 'long':
                    trailing_pnl = (position.stop_price - position.entry_price) * position.size
                else:  # short
                    trailing_pnl = (position.entry_price - position.stop_price) * position.size
                
                exit_conditions.append(ExitCondition(
                    exit_type=ExitType.TRAILING_STOP,
                    exit_price=position.stop_price,
                    pnl=trailing_pnl,
                    r_multiple=current_r,
                    exit_pct=100.0,
                    priority=1
                ))
            
            # Take profit levels (new multi-level system)
            if position.tp_levels:
                for tp_level in position.tp_levels:
                    if tp_level.get('executed', False):
                        continue
                    
                    tp_price = tp_level.get('price')
                    if tp_price is None:
                        continue
                    
                    tp_hit = False
                    if position.direction == 'long':
                        if current_price >= tp_price:
                            tp_hit = True
                    else:  # short
                        if current_price <= tp_price:
                            tp_hit = True
                    
                    if tp_hit:
                        target_r = tp_level.get('target_r', 3.0)
                        if position.direction == 'long':
                            tp_pnl = (tp_price - position.entry_price) * position.size
                        else:  # short
                            tp_pnl = (position.entry_price - tp_price) * position.size
                        
                        exit_conditions.append(ExitCondition(
                            exit_type=ExitType.TAKE_PROFIT,
                            exit_price=tp_price,
                            pnl=tp_pnl,
                            r_multiple=target_r,
                            exit_pct=tp_level.get('exit_pct', 100.0),
                            priority=2
                        ))
            # Legacy: single target_price
            elif position.target_price is not None:
                tp_hit = False
                if position.direction == 'long':
                    if current_price >= position.target_price:
                        tp_hit = True
                else:  # short
                    if current_price <= position.target_price:
                        tp_hit = True
                
                if tp_hit:
                    if position.direction == 'long':
                        tp_pnl = (position.target_price - position.entry_price) * position.size
                    else:  # short
                        tp_pnl = (position.entry_price - position.target_price) * position.size
                    
                    exit_conditions.append(ExitCondition(
                        exit_type=ExitType.TAKE_PROFIT,
                        exit_price=position.target_price,
                        pnl=tp_pnl,
                        r_multiple=3.0,  # Default
                        exit_pct=100.0,
                        priority=2
                    ))
            
            # Resolve simultaneous exits using ExitResolver
            if exit_conditions:
                best_exit = self.exit_resolver.resolve(exit_conditions)
                if best_exit:
                    # Determine exit reason and price
                    if best_exit.exit_type == ExitType.HARD_STOP:
                        exit_reason = 'stop_force'
                        close_price = best_exit.exit_price  # Use stop price
                    elif best_exit.exit_type == ExitType.TRAILING_STOP:
                        exit_reason = 'trailing_stop'
                        close_price = best_exit.exit_price
                    elif best_exit.exit_type == ExitType.TAKE_PROFIT:
                        exit_reason = 'target'
                        close_price = best_exit.exit_price
                    else:
                        exit_reason = 'unknown'
                        close_price = current_price
                    
                    # Handle partial TP exits
                    if best_exit.exit_pct < 100.0:
                        # Partial TP exit
                        exit_size = position.size * (best_exit.exit_pct / 100.0)
                        self._execute_partial_exit(position, close_price, current_time, exit_size)
                        # Mark TP level as executed
                        for tp_level in position.tp_levels:
                            if abs(tp_level.get('price', 0) - best_exit.exit_price) < 0.0001:
                                tp_level['executed'] = True
                                break
                    else:
                        # Full exit
                        positions_to_close.append((pos_idx, exit_reason))
                        continue
        
        # Close positions (in reverse order to maintain indices)
        for pos_idx, exit_reason in reversed(positions_to_close):
            if pos_idx < len(self.positions):
                position = self.positions[pos_idx]
                # If stop was breached intrabar, close at stop price to cap loss
                close_price = position.stop_price if exit_reason == 'stop_force' else current_price
                self._close_position(position, current_time, close_price, exit_reason)
                self.positions.pop(pos_idx)
    
    def _process_entries(
        self,
        current_time: pd.Timestamp,
        current_price: float,
        signals: pd.DataFrame,
        df_by_tf: Dict[str, pd.DataFrame]
    ) -> None:
        """
        Process entry signals and pending signals.
        
        Respects stop_signal_search_minutes_before by filtering out new signals
        that arrive after the stop time.
        """
        """Process entry signals.
        
        Args:
            current_time: Current bar timestamp
            current_price: Current market price
            signals: Signals DataFrame
            df_by_tf: Multi-timeframe data
        """
        # Initialize day tracking (safe when called outside run())
        self._ensure_day_initialized(current_time)

        # Enforce max daily loss cap: stop taking new trades for the rest of the day
        if self._daily_loss_limit_reached(current_time):
            self._daily_loss_rejections += 1
            # Drop any waiting signals; they should not carry past a forced daily-stop.
            if hasattr(self, 'pending_signals'):
                self.pending_signals = []
            return

        # STEP 3: Check for randomized entry context (force entry when enabled)
        # This bypasses normal signal validation but uses identical execution path
        if self.randomized_entry_context.enabled:
            # Check if this is the randomized entry bar (using stored bar index)
            if (hasattr(self, '_current_bar_index') and
                self.randomized_entry_context.entry_bar_index is not None and 
                self._current_bar_index == self.randomized_entry_context.entry_bar_index and
                self.randomized_entry_context.entry_price is not None and
                self.randomized_entry_context.direction is not None):
                
                # Get entry timeframe for stop calculation
                entry_tf = self.strategy.config.timeframes.entry_tf
                df_entry = df_by_tf.get(entry_tf)
                if df_entry is not None and len(df_entry) > 0:
                        
                        # Check if we can take another position
                        exec_cfg = getattr(self.strategy.config, 'execution', None)
                        max_positions = getattr(exec_cfg, 'max_positions', 1) if exec_cfg else 1
                        
                        # Check max trades per day limit
                        if self.max_trades_per_day is not None:
                            current_date = current_time.normalize()
                            trades_today = self._trades_per_day.get(current_date, 0)
                            if trades_today >= self.max_trades_per_day:
                                return  # Daily limit reached
                        
                        if len(self.positions) < max_positions:
                            # Get entry bar for stop calculation
                            entry_bars = df_entry[df_entry.index <= current_time]
                            if len(entry_bars) > 0:
                                entry_row = entry_bars.iloc[-1]
                                
                                # Use randomized entry price (NOT bar close)
                                entry_price = self.randomized_entry_context.entry_price
                                direction = self.randomized_entry_context.direction
                                dir_int = 1 if direction == 'long' else -1
                                
                                # CRITICAL: Recalculate stop AFTER randomization (uses entry_price, not bar.close)
                                decision_bar_time = entry_bars.index[-1]
                                decision_bar_idx = df_entry.index.get_loc(decision_bar_time)
                                stop_price = self._get_stop_price_with_fallback(
                                    entry_row=entry_row,
                                    dir_int=dir_int,
                                    reference_price=float(entry_price),
                                    df_entry=df_entry,
                                    decision_bar_idx=int(decision_bar_idx),
                                )
                                
                                # Validate stop
                                if stop_price is not None and ((dir_int == 1 and stop_price < entry_price) or (dir_int == -1 and stop_price > entry_price)):
                                    # Create synthetic signal for entry execution
                                    synthetic_signal = {
                                        'direction': direction,
                                        'entry_price': entry_price,
                                        'stop_price': stop_price,
                                        'immediate_execution': True  # Force immediate execution
                                    }
                                    
                                    # Calculate position size (AFTER stop is recalculated)
                                    risk_config = self.strategy.config.risk
                                    risk_pct = risk_config.risk_per_trade_pct
                                    quantity = self._calculate_position_size(entry_price, stop_price, risk_pct, current_time=current_time)
                                    
                                    if quantity > 0:
                                        # Force entry execution (bypasses should_enter check)
                                        try:
                                            entry_result = self._enter_position(synthetic_signal, current_time, current_price, quantity)
                                            if entry_result is True:
                                                # Successfully entered - clear randomized entry context for this iteration
                                                # (will be reset for next MC iteration)
                                                return
                                        except Exception as e:
                                            self.logger.debug(f"Randomized entry execution exception: {e}")
        
        # Push new signals for this timestamp into pending queue
        # Check for signals that occurred at or before current_time (not just exact match)
        # This handles cases where signal TF (2h) and entry TF (1h) timestamps don't align exactly
        if len(signals) > 0:
            # Find signals with timestamp <= current_time that haven't been processed yet
            available_signals = signals[signals.index <= current_time]
            if len(available_signals) > 0:
                # Track which signal timestamps we've already added
                if not hasattr(self, '_processed_signal_times'):
                    self._processed_signal_times = set()
                
                for sig_time in available_signals.index:
                    if sig_time not in self._processed_signal_times and not self._should_stop_signal_search(sig_time):
                        self._processed_signal_times.add(sig_time)
                        sig = available_signals.loc[sig_time]
                        
                        # Handle both single signal (Series) and multiple signals (DataFrame) cases
                        if isinstance(sig, pd.DataFrame):
                            new_sigs = [row for _, row in sig.iterrows()]
                        else:
                            new_sigs = [sig]
                        
                        for s in new_sigs:
                            # Check if signal is marked for immediate execution (master filters OFF)
                            immediate_execution = s.get('immediate_execution', False)
                            
                            if immediate_execution:
                                # Master filters OFF: Entry conditions already checked in generate_signals
                                # Execute immediately using current entry bar (matches NinjaTrader behavior)
                                # If we can't execute (already in position, etc.), discard signal (no waiting)
                                
                                # Track execution failures for debugging
                                if not hasattr(self, '_immediate_exec_failures'):
                                    self._immediate_exec_failures = {'already_in_position': 0, 'no_entry_bars': 0, 
                                                                     'invalid_stop': 0, 'zero_quantity': 0, 
                                                                     'execution_failed': 0}
                                
                                # Check if we can take another position (respect max_positions limit)
                                exec_cfg = getattr(self.strategy.config, 'execution', None)
                                max_positions = getattr(exec_cfg, 'max_positions', 1) if exec_cfg else 1
                                
                                if len(self.positions) < max_positions:  # Allow multiple positions up to max_positions
                                    entry_tf = self.strategy.config.timeframes.entry_tf
                                    df_entry = df_by_tf.get(entry_tf)
                                    if df_entry is not None and len(df_entry) > 0:
                                        # Use current entry bar (matches NT: uses current Entry TF bar when CheckForSignals is called)
                                        entry_bars = df_entry[df_entry.index <= current_time]
                                        if len(entry_bars) > 0:
                                            entry_row = entry_bars.iloc[-1]
                                            decision_bar_time = entry_bars.index[-1]
                                            decision_bar_idx = df_entry.index.get_loc(decision_bar_time)
                                            
                                            dir_int = 1 if s['direction'] == 'long' else -1
                                            
                                            # Recalculate prices using current entry bar
                                            decision_close = float(entry_row['close'])
                                            stop_price = self._get_stop_price_with_fallback(
                                                entry_row=entry_row,
                                                dir_int=dir_int,
                                                reference_price=float(decision_close),
                                                df_entry=df_entry,
                                                decision_bar_idx=int(decision_bar_idx),
                                            )
                                            
                                            # Validate stop
                                            if stop_price is not None and ((dir_int == 1 and stop_price < decision_close) or (dir_int == -1 and stop_price > decision_close)):
                                                fill = self._get_entry_fill(df_entry, decision_bar_idx, decision_bar_time, decision_close)
                                                if fill is None:
                                                    self._immediate_exec_failures['no_entry_bars'] += 1
                                                    continue
                                                fill_time, fill_price = fill

                                                # Check max trades per day limit based on FILL date
                                                if self.max_trades_per_day is not None:
                                                    fill_date = fill_time.normalize()
                                                    trades_today = self._trades_per_day.get(fill_date, 0)
                                                    if trades_today >= self.max_trades_per_day:
                                                        if 'daily_limit' not in self._immediate_exec_failures:
                                                            self._immediate_exec_failures['daily_limit'] = 0
                                                        self._immediate_exec_failures['daily_limit'] += 1
                                                        continue

                                                # If the next open gaps beyond the stop, skip the trade
                                                if (dir_int == 1 and fill_price <= stop_price) or (dir_int == -1 and fill_price >= stop_price):
                                                    self._immediate_exec_failures['invalid_stop'] += 1
                                                    continue

                                                # Update signal with current prices (create copy to avoid SettingWithCopyWarning)
                                                s = s.copy() if hasattr(s, 'copy') else dict(s)
                                                # Engine owns execution: fill price/time are derived from market data
                                                s['entry_price'] = float(fill_price)
                                                s['stop_price'] = stop_price
                                                
                                                # Calculate position size
                                                risk_config = self.strategy.config.risk
                                                risk_pct = risk_config.risk_per_trade_pct
                                                quantity = self._calculate_position_size(decision_close, stop_price, risk_pct, current_time=fill_time)
                                                
                                                if quantity > 0:
                                                    # Try to execute immediately
                                                    try:
                                                        entry_result = self._enter_position(s, fill_time, float(fill_price), quantity)
                                                        if entry_result is True:
                                                            # Successfully entered - signal executed, discard (don't add to pending)
                                                            continue
                                                        else:
                                                            self._immediate_exec_failures['execution_failed'] += 1
                                                    except Exception as e:
                                                        self._immediate_exec_failures['execution_failed'] += 1
                                                        if self._immediate_exec_failures['execution_failed'] <= 3:
                                                            self.logger.debug(f"Immediate execution exception: {e}")
                                                else:
                                                    self._immediate_exec_failures['zero_quantity'] += 1
                                            else:
                                                self._immediate_exec_failures['invalid_stop'] += 1
                                        else:
                                            self._immediate_exec_failures['no_entry_bars'] += 1
                                    else:
                                        self._immediate_exec_failures['no_entry_bars'] += 1
                                else:
                                    self._immediate_exec_failures['already_in_position'] += 1
                                
                                # If we get here, immediate execution failed or we're already in position
                                # Discard signal (matches NinjaTrader: no waiting signal when master filters OFF)
                                continue
                            
                            # Master filters ON: Add to pending queue for entry condition checking on subsequent bars
                            self.pending_signals.append({
                                'signal': s,
                                'created_at': sig_time,
                                'bars_waited': 0
                            })
        
        if len(self.pending_signals) == 0:
            return
        
        # Check if we can take another position (respect max_positions limit)
        exec_cfg = getattr(self.strategy.config, 'execution', None)
        max_positions = getattr(exec_cfg, 'max_positions', 1) if exec_cfg else 1
        
        if len(self.positions) >= max_positions:
            return  # Already at max positions, skip processing pending signals
        
        entry_tf = self.strategy.config.timeframes.entry_tf
        df_entry = df_by_tf.get(entry_tf)
        if df_entry is None or len(df_entry) == 0:
            return
        
        # Config for waiting logic
        exec_cfg = getattr(self.strategy.config, 'execution', None)
        max_wait_bars = getattr(exec_cfg, 'max_wait_bars', None)
        
        still_pending = []
        rejected_count = {'expired': 0, 'no_entry_bars': 0, 'ema_alignment': 0, 'macd_progression': 0, 
                         'ema_gating': 0, 'invalid_stop': 0, 'zero_quantity': 0, 'cannot_afford': 0}
        
        for pending in self.pending_signals:
            pending['bars_waited'] += 1
            
            # Expire if exceeded max_wait_bars (when configured)
            if max_wait_bars is not None and max_wait_bars > 0 and pending['bars_waited'] > max_wait_bars:
                rejected_count['expired'] += 1
                continue
            
            signal = pending['signal']
            direction = signal['direction']
            signal_created_at = pending['created_at']
            
            # Find current entry row
            # CRITICAL FIX: Use exact match or most recent bar to avoid processing same signal multiple times
            entry_bars = df_entry[df_entry.index <= current_time]
            if len(entry_bars) == 0:
                still_pending.append(pending)
                rejected_count['no_entry_bars'] += 1
                continue
            
            # Use the most recent bar (last one)
            entry_idx = entry_bars.index[-1]
            entry_row = entry_bars.iloc[-1]
            decision_bar_idx = df_entry.index.get_loc(entry_idx)
            
            # CRITICAL FIX: Ensure we only process each bar once per signal
            # If signal was created at a different time, use the current bar's timestamp
            if entry_idx != current_time and entry_idx in getattr(self, '_processed_entry_bars', set()):
                # This bar was already processed for this signal, skip
                still_pending.append(pending)
                continue
            
            # Track processed bars
            if not hasattr(self, '_processed_entry_bars'):
                self._processed_entry_bars = set()
            self._processed_entry_bars.add(entry_idx)
            
            dir_int = 1 if direction == 'long' else -1
            
            # Re-check entry-timeframe conditions (EMA alignment + MACD progression + optional EMA)
            # NOTE: Signals are created when signal TF conditions are met, then we check entry conditions
            # on each entry TF bar until they're met (matches NinjaTrader behavior)
            entry_ma_cfg = self.strategy.config.ma_settings.get('entry', None)
            entry_macd_cfg = self.strategy.config.macd_settings.get('entry', None)
            if entry_ma_cfg is None or entry_macd_cfg is None:
                continue
            
            # Check EMA alignment on current entry bar
            if not self.strategy._check_ema_alignment(entry_row, entry_ma_cfg, dir_int):
                still_pending.append(pending)
                rejected_count['ema_alignment'] += 1
                continue
            
            # Check MACD progression on current entry bar
            entry_df_idx = df_entry.index.get_loc(entry_idx)
            if not self.strategy._check_macd_progression(df_entry, entry_df_idx, dir_int, entry_macd_cfg.min_bars):
                still_pending.append(pending)
                rejected_count['macd_progression'] += 1
                continue
            
            # Optional EMA gating (slowest vs optional) if enabled
            if entry_ma_cfg.use_optional and entry_ma_cfg.optional > 0 and entry_ma_cfg.use_slowest:
                slowest_name = f'sma{entry_ma_cfg.slowest}'
                opt_name = f'ema{entry_ma_cfg.optional}'
                if slowest_name in entry_row and opt_name in entry_row:
                    slowest_val = entry_row[slowest_name]
                    opt_val = entry_row[opt_name]
                    if pd.notna(slowest_val) and pd.notna(opt_val):
                        if dir_int == 1 and slowest_val <= opt_val:
                            still_pending.append(pending)
                            rejected_count['ema_gating'] += 1
                            continue
                        if dir_int == -1 and slowest_val >= opt_val:
                            still_pending.append(pending)
                            rejected_count['ema_gating'] += 1
                            continue
            
            # Recalculate stop loss and entry price based on current entry bar
            # (These may have changed since signal generation)
            decision_close = float(entry_row['close'])
            stop_price = self._get_stop_price_with_fallback(
                entry_row=entry_row,
                dir_int=dir_int,
                reference_price=float(decision_close),
                df_entry=df_entry,
                decision_bar_idx=int(decision_bar_idx),
            )
            
            # Check if stop price is valid (not None)
            if stop_price is None:
                rejected_count['invalid_stop'] += 1
                still_pending.append(pending)
                continue
            
            # Validate stop side
            # CRITICAL FIX: For longs, stop must be BELOW entry (stop_price < entry_price)
            # For shorts, stop must be ABOVE entry (stop_price > entry_price)
            if dir_int == 1 and stop_price >= decision_close:
                rejected_count['invalid_stop'] += 1
                # Log first few invalid stops
                if rejected_count['invalid_stop'] <= 5:
                    sl_ma_val = entry_row.get('sl_ma', 'N/A')
                    self.logger.debug(f"invalid_stop LONG - entry={entry_price:.2f}, stop={stop_price:.2f}, "
                                     f"sl_ma={sl_ma_val}, buffer={self.strategy.config.stop_loss.buffer_pips} "
                                     f"points, diff={entry_price - stop_price:.2f}")
                continue
            if dir_int == -1 and stop_price <= decision_close:
                rejected_count['invalid_stop'] += 1
                # Log first few invalid stops
                if rejected_count['invalid_stop'] <= 5:
                    sl_ma_val = entry_row.get('sl_ma', 'N/A')
                    self.logger.debug(f"invalid_stop SHORT - entry={entry_price:.2f}, stop={stop_price:.2f}, "
                                     f"sl_ma={sl_ma_val}, buffer={self.strategy.config.stop_loss.buffer_pips} "
                                     f"points, diff={stop_price - entry_price:.2f}")
                continue

            fill = self._get_entry_fill(df_entry, decision_bar_idx, entry_idx, decision_close)
            if fill is None:
                rejected_count['no_entry_bars'] += 1
                continue
            fill_time, fill_price = fill

            # Enforce max trades/day based on the actual fill date
            if self.max_trades_per_day is not None:
                fill_day = fill_time.normalize()
                trades_today = self._trades_per_day.get(fill_day, 0)
                if trades_today >= self.max_trades_per_day:
                    if not hasattr(self, '_daily_limit_rejections'):
                        self._daily_limit_rejections = 0
                    self._daily_limit_rejections += 1
                    continue

            # If the fill open gaps beyond the stop, skip (would be immediately stopped)
            if (dir_int == 1 and fill_price <= stop_price) or (dir_int == -1 and fill_price >= stop_price):
                rejected_count['invalid_stop'] += 1
                continue
            
            # Position sizing
            risk_config = self.strategy.config.risk
            risk_pct = risk_config.risk_per_trade_pct
            quantity = self._calculate_position_size(decision_close, stop_price, risk_pct, current_time=fill_time)
            if quantity <= 0:
                rejected_count['zero_quantity'] += 1
                # Log why quantity is zero with detailed diagnostics
                if rejected_count['zero_quantity'] <= 5:  # Log first 5 for better diagnostics
                    price_risk = abs(entry_price - stop_price)
                    if risk_config.sizing_mode == "account_size":
                        sizing_base = float(risk_config.account_size)
                    elif risk_config.sizing_mode == "daily_equity":
                        sizing_base = float(self._day_start_equity_by_day.get(fill_time.normalize(), self.account.equity))
                    else:
                        sizing_base = float(self.account.equity)
                    risk_amount = sizing_base * (risk_pct / 100.0)
                    # Get stop loss calculation details for diagnostics
                    sl_ma_val = entry_row.get('sl_ma', 'N/A')
                    sl_config = self.strategy.config.stop_loss
                    buffer_info = f"{sl_config.buffer_pips} {sl_config.buffer_unit}"
                    if sl_config.buffer_unit == "pip":
                        buffer_info += f" (pip_size={sl_config.pip_size})"
                    
                    # Calculate stop loss pips for forex
                    if self.market_spec.asset_class == 'forex' and self.market_spec.pip_value:
                        stop_loss_pips = price_risk / self.market_spec.pip_value
                        pip_info = f", stop_loss_pips={stop_loss_pips:.4f}"
                    else:
                        pip_info = ""
                    
                    self.logger.debug(f"zero_quantity - entry={entry_price:.5f}, stop={stop_price:.5f}, "
                                     f"price_risk={price_risk:.6f}{pip_info}, "
                                     f"sl_ma={sl_ma_val}, buffer={buffer_info}, "
                                     f"risk_amount=${risk_amount:.2f}, sizing_base=${sizing_base:.2f}, "
                                     f"direction={'long' if dir_int == 1 else 'short'}")
                continue
            
            # Update signal with current entry price and stop price before entering
            # Create a copy to avoid SettingWithCopyWarning
            signal_copy = signal.copy()
            # Engine owns execution: fill price/time are derived from market data
            signal_copy['entry_price'] = float(fill_price)
            signal_copy['stop_price'] = stop_price
            signal = signal_copy
            
            # Enter and clear the queue (single position model)
            try:
                entry_result = self._enter_position(signal, fill_time, float(fill_price), quantity)
                if entry_result is not True:  # None or False indicates failure
                    rejected_count['cannot_afford'] += 1
                    # Log why cannot_afford (only if margin checks are enabled)
                    if self.enforce_margin_checks and rejected_count['cannot_afford'] <= 5:
                        entry_price = signal['entry_price']
                        margin_required = self.broker.calculate_margin_required(entry_price, quantity, is_intraday=True)
                        commission = self.broker.calculate_commission(entry_price, quantity)
                        position_cost = margin_required + commission
                        self.logger.debug(f"cannot_afford - entry={entry_price:.2f}, qty={quantity:.2f}, "
                                         f"margin=${margin_required:.2f}, commission=${commission:.2f}, "
                                         f"cost=${position_cost:.2f}, cash=${self.account.cash:.2f}, "
                                         f"equity=${self.account.equity:.2f}")
                    elif not self.enforce_margin_checks:
                        # This shouldn't happen - log it as an error
                        if rejected_count['cannot_afford'] <= 3:
                            self.logger.error(f"_enter_position returned {entry_result} but enforce_margin_checks=False! "
                                            f"This should not happen. entry={signal['entry_price']:.2f}, qty={quantity:.2f}")
                    continue
            except Exception as e:
                # Catch any exceptions during entry
                rejected_count['cannot_afford'] += 1
                if rejected_count['cannot_afford'] <= 3:
                    self.logger.error(f"Exception in _enter_position: {e}")
                continue
            
            self.pending_signals = []
            return
        
        # Update cumulative rejection counts
        for key in self._cumulative_rejections:
            self._cumulative_rejections[key] += rejected_count[key]
        
        # Keep unfilled signals
        self.pending_signals = still_pending
    
    def _calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        risk_pct: float,
        current_time: Optional[pd.Timestamp] = None,
    ) -> float:
        """Calculate position size based on risk percentage.
        
        For forex: Uses lot-based calculation
        For other assets: Uses direct quantity calculation
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            risk_pct: Risk percentage (e.g., 0.5 for 0.5%)
            
        Returns:
            Position quantity
        """
        # Calculate risk amount based on sizing mode
        risk_config = self.strategy.config.risk
        if risk_config.sizing_mode == "account_size":
            # Fixed risk based on account_size (more realistic, recommended)
            account_size = risk_config.account_size or self.initial_capital
            risk_amount = account_size * (risk_pct / 100.0)
        elif risk_config.sizing_mode == "daily_equity":
            ts = current_time
            if ts is None:
                ts = self._current_day_key
            if ts is None:
                ts = pd.Timestamp.utcnow()
            day_key = self._day_key(pd.Timestamp(ts))
            base = float(self._day_start_equity_by_day.get(day_key, float(self.account.equity)))
            if base <= 0:
                return 0.0
            risk_amount = base * (risk_pct / 100.0)
        else:
            # Dynamic risk based on current equity (compounds gains)
            current_equity = self.account.equity
            if current_equity <= 0:
                return 0.0  # Cannot trade with zero or negative equity
            risk_amount = current_equity * (risk_pct / 100.0)

        # Portfolio-level open-risk cap (risk-to-stop)
        # If enabled, scale risk_amount down to fit remaining risk budget.
        portfolio_cfg = getattr(self.strategy.config, 'portfolio', None)
        max_open_risk_pct = getattr(portfolio_cfg, 'max_open_risk_pct', None) if portfolio_cfg else None
        portfolio_enabled = bool(getattr(portfolio_cfg, 'enabled', False)) if portfolio_cfg else False

        if portfolio_enabled and max_open_risk_pct is not None and max_open_risk_pct > 0:
            if risk_config.sizing_mode == "account_size":
                portfolio_base = risk_config.account_size or self.initial_capital
            elif risk_config.sizing_mode == "daily_equity":
                ts = current_time
                if ts is None:
                    ts = self._current_day_key
                if ts is None:
                    ts = pd.Timestamp.utcnow()
                day_key = self._day_key(pd.Timestamp(ts))
                portfolio_base = float(self._day_start_equity_by_day.get(day_key, float(self.account.equity)))
            else:
                portfolio_base = self.account.equity

            max_open_risk_amount = float(portfolio_base) * (float(max_open_risk_pct) / 100.0)
            current_open_risk_amount = self._calculate_open_risk_amount()
            remaining_risk_amount = max_open_risk_amount - current_open_risk_amount

            if remaining_risk_amount <= 0:
                return 0.0

            risk_amount = min(risk_amount, remaining_risk_amount)
        # Calculate price risk (stop loss distance)
        price_risk = abs(entry_price - stop_price)
        
        if price_risk == 0:
            return 0.0
        
        # NOTE: Slippage is applied at entry time, not during position sizing
        # This matches industry standard - position sizing is based on intended entry/stop,
        # slippage affects actual execution but not the risk calculation
        price_risk_with_slippage = price_risk
        
        # For forex, use lot-based calculation
        # CRITICAL: Lot size calculation is based on RISK, NOT leverage.
        # Leverage only affects margin (checked separately below).
        # Formula: Lot Size = Risk Amount / (Stop Loss Pips × Pip Value per Lot)
        if self.market_spec.asset_class == 'forex' and self.market_spec.contract_size is not None:
            pip_size = self.market_spec.pip_value or 0.0001
            stop_loss_pips = price_risk_with_slippage / pip_size
            
            # DEBUG: Log position sizing only for very tight stops (<2 pips)
            import warnings
            if stop_loss_pips < 2.0:
                if risk_config.sizing_mode == "account_size":
                    base_amount = float(risk_config.account_size)
                elif risk_config.sizing_mode == "daily_equity":
                    ts = current_time or self._current_day_key or pd.Timestamp.utcnow()
                    day_key = self._day_key(pd.Timestamp(ts))
                    base_amount = float(self._day_start_equity_by_day.get(day_key, self.account.equity))
                else:
                    base_amount = float(self.account.equity)
                warnings.warn(
                    f"Position sizing: entry={entry_price:.5f}, stop={stop_price:.5f}, "
                    f"price_risk={price_risk:.5f}, pip_size={pip_size}, "
                    f"stop_loss_pips={stop_loss_pips:.2f}, risk_amount=${risk_amount:.2f} "
                    f"(based on {risk_config.sizing_mode}, base=${base_amount:.2f})"
                )
            
            # Safety check: Require minimum stop loss distance to prevent oversized positions
            # For forex, require at least 1 pip stop loss (0.0001 for most pairs)
            min_stop_loss_pips = 1.0
            if stop_loss_pips < min_stop_loss_pips:
                return 0.0  # Stop loss too close, skip trade
            
            # Calculate position quantity using correct FX formula
            # Formula: quantity = risk_amount / (stop_loss_pips × pip_value_per_unit)
            # This ensures correct position sizing for forex pairs
            pip_value_per_unit = self.market_spec.calculate_pip_value_per_unit(entry_price)
            if pip_value_per_unit <= 0:
                return 0.0
            
            # Calculate quantity directly (more explicit than lot_size conversion)
            quantity = self.market_spec.calculate_quantity_from_risk(
                risk_amount,
                stop_loss_pips,
                entry_price
            )
            
            if quantity <= 0:
                return 0.0
            
            # Convert to lot size for validation
            lot_size = quantity / self.market_spec.contract_size
            
            # Check minimum lot size
            if lot_size < self.market_spec.min_trade_size:
                return 0.0
            
            # Safety check: Maximum reasonable lot size to prevent oversized positions
            # This catches calculation errors (e.g., if stop loss is calculated incorrectly)
            max_reasonable_lot_size = 10.0
            if lot_size > max_reasonable_lot_size:
                import warnings
                if risk_config.sizing_mode == "account_size":
                    base_amount = float(risk_config.account_size)
                elif risk_config.sizing_mode == "daily_equity":
                    ts = current_time or self._current_day_key or pd.Timestamp.utcnow()
                    day_key = self._day_key(pd.Timestamp(ts))
                    base_amount = float(self._day_start_equity_by_day.get(day_key, self.account.equity))
                else:
                    base_amount = float(self.account.equity)
                warnings.warn(
                    f"⚠️ Position size too large: lot_size={lot_size:.4f} lots ({quantity:,.0f} units), "
                    f"entry={entry_price:.5f}, stop={stop_price:.5f}, "
                    f"price_risk={price_risk:.5f}, stop_loss_pips={stop_loss_pips:.2f} pips, "
                    f"pip_value_per_unit=${pip_value_per_unit:.6f}, "
                    f"risk_amount=${risk_amount:.2f} (based on {risk_config.sizing_mode}, base=${base_amount:.2f}). "
                    f"Skipping trade to prevent oversized position."
                )
                return 0.0
        else:
            # For other asset classes (stocks, crypto, futures), direct quantity calculation
            # For futures: need to account for contract multiplier
            if self.market_spec.asset_class == 'futures' and self.market_spec.contract_size is not None:
                # CRITICAL: NT uses tickSize and pointValue for futures calculation
                # dollarRiskPerContract = (slDistance / tickSize) * pointValue
                # For MES: tickSize = 0.25, pointValue = $1.25 per tick
                # So: dollarRiskPerContract = (34.5 / 0.25) * 1.25 = 138 * 1.25 = $172.50
                #
                # Our equivalent: dollar_risk_per_contract = price_risk * contract_size
                # For MES: contract_size = 5.0 ($5 per point)
                # So: dollar_risk_per_contract = 34.5 * 5.0 = $172.50 ✓ (same result)
                #
                # Use price_risk_with_slippage to account for slippage affecting entry price
                dollar_risk_per_contract = price_risk_with_slippage * self.market_spec.contract_size
                if dollar_risk_per_contract <= 0:
                    return 0.0
                quantity = risk_amount / dollar_risk_per_contract

                # If portfolio cap is enabled, respect the cap strictly.
                # Futures cannot be scaled below 1 contract; if we can't afford 1 contract
                # within the remaining risk budget, skip the trade.
                if portfolio_enabled and max_open_risk_pct is not None and max_open_risk_pct > 0:
                    if quantity < 1.0:
                        return 0.0
                    quantity = float(np.floor(quantity))
                    quantity = max(1.0, quantity)
                else:
                    # Match NinjaTrader behavior (minimum 1 contract)
                    quantity = max(1.0, quantity)
            else:
                # For stocks/crypto: direct quantity calculation
                quantity = risk_amount / price_risk_with_slippage
        
        # SEPARATE STEP: Check if we can afford this position (margin check)
        # This is where leverage matters - margin = (price * quantity) / leverage
        # Lot size was already calculated above based on risk, independent of leverage.
        # NOTE: Margin checks can be disabled for benchmarking to match NinjaTrader's default behavior
        
        # Log margin enforcement status (first time only)
        if not hasattr(self, '_margin_debug_logged_calc'):
            self.logger.debug(f"_calculate_position_size - enforce_margin_checks = {self.enforce_margin_checks}")
            self._margin_debug_logged_calc = True
        
        if self.enforce_margin_checks:
            # CRITICAL: For futures, brokers use free_margin (equity - used_margin) for margin checks,
            # not just cash. This allows trading with unrealized profits.
            if self.market_spec.asset_class == 'futures':
                available_margin = self.account.free_margin
                can_afford, required_cash = self.broker.can_afford_position(
                    entry_price,
                    quantity,
                    self.account.cash,  # For commission
                    available_margin=available_margin  # For margin
                )
            else:
                can_afford, required_cash = self.broker.can_afford_position(
                    entry_price,
                    quantity,
                    self.account.cash
                )
                available_margin = self.account.cash
            
            if not can_afford:
                # Reduce quantity to fit available margin (leverage affects this reduction)
                if self.market_spec.asset_class == 'futures':
                    quantity = self.broker.adjust_quantity_for_cash(
                        entry_price,
                        quantity,
                        self.account.cash,  # For commission
                        available_margin=available_margin  # For margin
                    )
                else:
                    quantity = self.broker.adjust_quantity_for_cash(
                        entry_price,
                        quantity,
                        available_margin
                    )
                
                # Check minimum trade size after reduction
                if self.market_spec.asset_class == 'forex' and self.market_spec.contract_size is not None:
                    reduced_lot_size = quantity / self.market_spec.contract_size
                    if reduced_lot_size < self.market_spec.min_trade_size:
                        return 0.0
                elif self.market_spec.asset_class == 'futures':
                    # CRITICAL FIX: If margin checks are disabled and we can't afford 1 contract,
                    # return 0 instead of forcing 1 contract (which would be impossible)
                    if not self.enforce_margin_checks:
                        # Check if we can afford at least 1 contract
                        margin_for_one = self.broker.calculate_margin_required(entry_price, 1.0, is_intraday=True)
                        commission_for_one = self.broker.calculate_commission(entry_price, 1.0)
                        if self.market_spec.asset_class == 'futures':
                            if margin_for_one > self.account.free_margin or commission_for_one > self.account.cash:
                                return 0.0  # Cannot afford even 1 contract
                        else:
                            if margin_for_one + commission_for_one > self.account.cash:
                                return 0.0  # Cannot afford even 1 contract
                    # CRITICAL: NT uses Math.Max(1, quantity) - always trades at least 1 contract
                    # Even if it exceeds risk amount or after cash adjustment (when margin checks enabled).
                    # Match NT behavior: ensure minimum of 1 contract for futures
                    quantity = max(1.0, quantity)
        
        if quantity <= 0:
            return 0.0
        
        return round(quantity, self.market_spec.quantity_precision)

    def _calculate_open_risk_amount(self) -> float:
        """Calculate current open risk (risk-to-stop) across all open positions.

        Open risk is defined as the loss incurred if each position is stopped at its
        current stop_price (ignoring slippage/fees for sizing purposes).
        """
        total = 0.0
        for pos in getattr(self, 'positions', []) or []:
            try:
                total += float(self._calculate_position_open_risk_amount(pos))
            except Exception:
                # If a position is malformed, be conservative and ignore it rather than crash sizing.
                continue
        return float(total)

    def _calculate_position_open_risk_amount(self, pos: 'Position') -> float:
        """Estimate open risk amount (cash units) for a single position."""
        entry_price = float(pos.entry_price)
        stop_price = float(pos.stop_price)
        quantity = float(pos.size)
        price_risk = abs(entry_price - stop_price)
        if price_risk <= 0 or quantity == 0:
            return 0.0

        if self.market_spec.asset_class == 'forex' and self.market_spec.contract_size is not None:
            pip_size = self.market_spec.pip_value or 0.0001
            stop_loss_pips = price_risk / pip_size
            pip_value_per_unit = self.market_spec.calculate_pip_value_per_unit(entry_price)
            if stop_loss_pips <= 0 or pip_value_per_unit <= 0:
                return 0.0
            return abs(quantity) * stop_loss_pips * pip_value_per_unit

        if self.market_spec.asset_class == 'futures' and self.market_spec.contract_size is not None:
            return abs(quantity) * price_risk * self.market_spec.contract_size

        # Stocks/crypto
        return abs(quantity) * price_risk

    def _get_stop_price_with_fallback(
        self,
        entry_row: pd.Series,
        dir_int: int,
        reference_price: float,
        df_entry: pd.DataFrame,
        decision_bar_idx: int,
    ) -> Optional[float]:
        """Get stop loss price, optionally falling back to ATR-based stop.

        Strategy stop is attempted first. If missing/invalid and portfolio stop
        fallback is enabled, compute fallback stop as:

        - long:  reference_price - (ATR(period) * multiplier)
        - short: reference_price + (ATR(period) * multiplier)
        """
        stop_price = self.strategy._calculate_stop_loss(entry_row, dir_int)
        if self._is_valid_stop(stop_price, reference_price, dir_int):
            return float(stop_price)

        portfolio_cfg = getattr(self.strategy.config, 'portfolio', None)
        if not portfolio_cfg or not getattr(portfolio_cfg, 'enabled', False):
            return None
        if not getattr(portfolio_cfg, 'stop_fallback_enabled', False):
            return None

        atr_period = int(getattr(portfolio_cfg, 'stop_fallback_atr_period', 14))
        atr_mult = float(getattr(portfolio_cfg, 'stop_fallback_atr_multiplier', 3.0))

        atr = self._calculate_atr_value(df_entry, decision_bar_idx, atr_period)
        if atr is None or not np.isfinite(atr) or atr <= 0:
            return None

        dist = atr_mult * atr
        fallback_stop = float(reference_price) - (float(dir_int) * dist)
        if self._is_valid_stop(fallback_stop, reference_price, dir_int):
            return float(fallback_stop)
        return None

    @staticmethod
    def _is_valid_stop(stop_price: Optional[float], reference_price: float, dir_int: int) -> bool:
        if stop_price is None:
            return False
        try:
            sp = float(stop_price)
        except (ValueError, TypeError):
            return False
        if not np.isfinite(sp):
            return False
        if dir_int == 1:
            return sp < float(reference_price)
        return sp > float(reference_price)

    @staticmethod
    def _calculate_atr_value(df: pd.DataFrame, end_idx: int, period: int) -> Optional[float]:
        """Compute ATR(period) at df.iloc[end_idx] using only past data (no lookahead)."""
        if df is None or len(df) == 0:
            return None
        if period <= 0:
            return None

        end_idx = int(end_idx)
        if end_idx <= 0:
            return None

        required_cols = {'high', 'low', 'close'}
        if not required_cols.issubset(df.columns):
            return None

        start_idx = max(0, end_idx - period - 1)
        window = df.iloc[start_idx:end_idx + 1]
        if len(window) < period + 1:
            return None

        high = window['high'].astype(float)
        low = window['low'].astype(float)
        close = window['close'].astype(float)
        prev_close = close.shift(1)

        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # First row has no prev_close
        tr = tr.iloc[1:]
        if len(tr) < period:
            return None

        atr = tr.iloc[-period:].mean()
        if pd.isna(atr):
            return None
        return float(atr)
    
    def _enter_position(
        self,
        signal: pd.Series,
        current_time: pd.Timestamp,
        current_price: float,
        quantity: float
    ) -> Optional[bool]:
        """Enter a new position.
        
        Args:
            signal: Signal Series with direction, entry_price, stop_price
            current_time: Entry timestamp
            current_price: Current market price
            quantity: Position quantity
        """
        direction = signal['direction']
        entry_price = float(signal['entry_price'])
        stop_price = float(signal['stop_price'])
        target_price = None
        if 'target_price' in signal and signal['target_price'] is not None:
            try:
                target_price = float(signal['target_price'])
            except (ValueError, TypeError):
                target_price = None
        
        # Apply slippage and commission
        entry_price_adj = self.broker.apply_slippage(entry_price, quantity, is_entry=True)
        entry_commission = self.broker.calculate_commission(entry_price_adj, quantity)
        
        # Calculate margin required
        margin_required = self.broker.calculate_margin_required(entry_price_adj, quantity, is_intraday=True)
        
        # Check margin requirements (can be disabled for benchmarking to match NT default)
        # Log margin enforcement status (first time only)
        if not hasattr(self, '_margin_debug_logged'):
            self.logger.debug(f"_enter_position - enforce_margin_checks = {self.enforce_margin_checks}")
            self._margin_debug_logged = True
        
        if self.enforce_margin_checks:
            # CRITICAL: For futures, check free_margin for margin, cash for commission
            # For other assets, check cash for both
            if self.market_spec.asset_class == 'futures':
                # Futures: margin is locked but not deducted from cash
                # Commission IS deducted from cash
                if margin_required > self.account.free_margin:
                    # Log why rejected
                    if not hasattr(self, '_margin_reject_debug_count'):
                        self._margin_reject_debug_count = 0
                    if self._margin_reject_debug_count < 3:
                        self.logger.debug(f"_enter_position REJECTED - margin_required=${margin_required:.2f} > free_margin=${self.account.free_margin:.2f}")
                        self._margin_reject_debug_count += 1
                    return None  # Cannot afford margin
                if entry_commission > self.account.cash:
                    # Log why rejected
                    if not hasattr(self, '_commission_reject_debug_count'):
                        self._commission_reject_debug_count = 0
                    if self._commission_reject_debug_count < 3:
                        self.logger.debug(f"_enter_position REJECTED - commission=${entry_commission:.2f} > cash=${self.account.cash:.2f}")
                        self._commission_reject_debug_count += 1
                    return None  # Cannot afford commission
            else:
                # Other assets: check total cost against cash
                position_cost = margin_required + entry_commission
                if position_cost > self.account.cash:
                    return None  # Cannot afford (return None to indicate failure)
        
        # Deduct commission only; margin is reserved (used_margin) but not removed from equity
        self.account.cash -= entry_commission
        self.account.commission_paid += entry_commission
        slippage_price = self.broker._slippage_price_units()
        self.account.slippage_paid += slippage_price * abs(quantity)
        
        # Calculate R value (stop loss distance)
        r_value = abs(entry_price_adj - stop_price)
        
        # Calculate partial exit prices and trailing start price using trade management config
        partial_prices = []  # Support multiple partial levels
        trailing_start_price = None
        
        # Get trade management configs
        trailing_config_dict = self.trade_mgmt.get_trailing_stop_config()
        partial_config_dict = self.trade_mgmt.get_partial_exit_config()
        tp_config_dict = self.trade_mgmt.get_take_profit_config()
        
        # Calculate partial exit prices (support multiple levels)
        if partial_config_dict.get('enabled', False):
            levels = partial_config_dict.get('levels', [])
            # Legacy support: if levels empty but level_r exists, create single level
            if not levels and partial_config_dict.get('level_r') is not None:
                levels = [{
                    'type': 'r_based',
                    'level_r': partial_config_dict.get('level_r'),
                    'exit_pct': partial_config_dict.get('exit_pct', 80.0)
                }]
            
            for level in levels:
                if not level.get('enabled', True):
                    continue
                
                level_type = level.get('type', 'r_based')
                if level_type == 'r_based' and level.get('level_r') is not None:
                    level_r = level.get('level_r')
                    if direction == 'long':
                        partial_price = entry_price_adj + (level_r * r_value)
                    else:  # short
                        partial_price = entry_price_adj - (level_r * r_value)
                    partial_prices.append({
                        'price': partial_price,
                        'exit_pct': level.get('exit_pct', 50.0),
                        'level_r': level_r
                    })
                # TODO: Support other partial types (ATR, price, percentage, time)
        
        # Calculate trailing start price
        if trailing_config_dict.get('enabled', False):
            activation_type = trailing_config_dict.get('activation_type', 'r_based')
            if activation_type == 'r_based':
                activation_r = trailing_config_dict.get('activation_r', 0.5)
                if direction == 'long':
                    trailing_start_price = entry_price_adj + (activation_r * r_value)
                else:  # short
                    trailing_start_price = entry_price_adj - (activation_r * r_value)
            # TODO: Support other activation types (ATR, price, time)
        
        # Store first partial price for backward compatibility (legacy code expects single partial_price)
        partial_price = partial_prices[0]['price'] if partial_prices else None
        
        # Create position
        position = Position(
            instrument=self.market_spec.symbol,
            direction=direction,
            size=quantity,
            entry_price=entry_price_adj,
            stop_price=stop_price,
            target_price=target_price,
            open_time=current_time,
            entry_commission=entry_commission,
            margin_used=margin_required,
            initial_size=quantity,
            # Initialize MFE/MAE with current entry state
            max_favorable_price=entry_price_adj,
            max_adverse_price=entry_price_adj,
            max_favorable_pnl=0.0,
            max_adverse_pnl=0.0,
            partial_taken=False,
            partial_price=partial_price,  # Legacy support
            partial_levels=partial_prices,  # NEW: Multiple partial levels
            trailing_active=False,
            trailing_start_price=trailing_start_price,
            last_trail_time=None,
            last_notified_r=0.0,
            r_value=r_value,
            tp_levels=[]  # Will be calculated if take profit enabled
        )
        
        # Calculate take profit levels if enabled
        if tp_config_dict.get('enabled', False):
            levels = tp_config_dict.get('levels', [])
            # Legacy support: if levels empty but target_r exists, create single level
            if not levels and tp_config_dict.get('target_r') is not None:
                levels = [{
                    'type': 'r_based',
                    'target_r': tp_config_dict.get('target_r'),
                    'exit_pct': 100.0
                }]
            
            for level in levels:
                if not level.get('enabled', True):
                    continue
                
                level_type = level.get('type', 'r_based')
                if level_type == 'r_based' and level.get('target_r') is not None:
                    target_r = level.get('target_r')
                    if direction == 'long':
                        tp_price = entry_price_adj + (target_r * r_value)
                    else:  # short
                        tp_price = entry_price_adj - (target_r * r_value)
                    position.tp_levels.append({
                        'price': tp_price,
                        'exit_pct': level.get('exit_pct', 100.0),
                        'target_r': target_r,
                        'executed': False
                    })
            # Also set target_price for backward compatibility (first TP level)
            if position.tp_levels:
                position.target_price = position.tp_levels[0]['price']
        
        self.positions.append(position)

        # Track entries per day for max_trades_per_day limit
        if self.max_trades_per_day is not None:
            entry_day = current_time.normalize()
            self._trades_per_day[entry_day] = self._trades_per_day.get(entry_day, 0) + 1
        
        # Return True to indicate successful entry
        return True
    
    def _close_position(
        self,
        position: Position,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str
    ) -> None:
        """Close a position and create Trade record.
        
        Args:
            position: Position to close
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        # Apply slippage and commission
        exit_price_adj = self.broker.apply_slippage(exit_price, position.size, is_entry=False)
        exit_commission = self.broker.calculate_commission(exit_price_adj, position.size)
        
        # Calculate realized P&L
        # NOTE: entry_price and exit_price_adj already include slippage
        # So realized_pnl already accounts for slippage
        realized_pnl = self.broker.calculate_realized_pnl(
            position.entry_price,
            exit_price_adj,
            position.size,
            position.direction
        )
        
        # Calculate R multiple
        initial_risk = abs(position.entry_price - position.stop_price)
        # Check both initial_risk and position.size to avoid division by zero
        if initial_risk > 0 and abs(position.size) > 0:
            r_multiple = realized_pnl / (initial_risk * abs(position.size))
        else:
            r_multiple = 0.0
        
        # Calculate total costs for trade record
        total_commission = position.entry_commission + exit_commission
        # Slippage is already included in realized_pnl via price adjustments
        # But we track it separately for reporting
        slippage_price = self.broker._slippage_price_units()
        total_slippage = slippage_price * abs(position.size) * 2  # Entry + exit
        
        # MONTE CARLO COMPATIBLE: Track equity before trade
        # CRITICAL: Use the LAST recorded equity (after previous trade), not current account equity
        # This ensures returns are calculated against the sequential equity path, not account equity
        # which may include unrealized P&L from other positions or bar-by-bar updates
        if len(self.equity_list) > 0:
            equity_before = self.equity_list[-1]  # Last recorded equity after previous trade
        else:
            equity_before = self.initial_capital  # First trade
        
        # Cash accounting
        # On entry: cash -= entry_commission
        # On exit: cash += realized_pnl - exit_commission
        # Margin was never deducted from cash; it only reduces free margin
        exit_proceeds = realized_pnl - exit_commission
        self.account.cash += exit_proceeds
        self.account.commission_paid += exit_commission
        self.account.slippage_paid += slippage_price * abs(position.size)
        
        # MONTE CARLO COMPATIBLE: Calculate P&L after all costs
        # pnl_after_costs = realized_pnl - total_commission - total_slippage
        # Note: slippage is already in realized_pnl via price adjustments,
        # but we track it separately for reporting
        pnl_after_costs = realized_pnl - total_commission
        
        # MONTE CARLO COMPATIBLE: Calculate trade return BEFORE updating equity
        # return_i = pnl_after_costs_i / equity_before_trade_i
        # This return is based on the equity from the previous trade, ensuring permutation-invariance
        if equity_before > 0:
            trade_return = pnl_after_costs / equity_before
        else:
            trade_return = 0.0
        
        # MONTE CARLO COMPATIBLE: Calculate equity after trade
        # CRITICAL: Use simple addition (equity_before + pnl_after_costs) for proper compounding
        # This ensures the equity path is built sequentially from returns, not from account equity
        equity_after = equity_before + pnl_after_costs
        
        # CRITICAL FIX: Populate partial_exits from position's partial_exits list
        # Convert PartialExit dataclasses to dictionaries for Trade record
        partial_exits_list = []
        if hasattr(position, 'partial_exits') and position.partial_exits:
            for pe in position.partial_exits:
                partial_exits_list.append({
                    'exit_time': pe.exit_time,
                    'exit_price': pe.exit_price,
                    'quantity': pe.quantity,
                    'pnl': pe.pnl
                })
        
        # Calculate MFE/MAE in R-multiples
        mfe_r = None
        mae_r = None
        if position.r_value > 0 and abs(position.initial_size) > 0:
            if position.max_favorable_pnl != 0:
                mfe_r = position.max_favorable_pnl / (position.r_value * abs(position.initial_size))
            if position.max_adverse_pnl != 0:
                mae_r = position.max_adverse_pnl / (position.r_value * abs(position.initial_size))
        
        # Create trade record with Monte Carlo compatible structure
        trade = Trade(
            entry_time=position.open_time,
            exit_time=exit_time,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price_adj,
            quantity=position.size,  # REQUIRED field
            pnl_raw=realized_pnl,  # Raw P&L before costs
            pnl_after_costs=pnl_after_costs,  # REQUIRED for MC returns
            commission=total_commission,
            slippage=total_slippage,
            stop_price=position.stop_price,
            exit_reason=exit_reason,
            partial_exits=partial_exits_list,  # FIXED: Now populated with actual partial exit records
            mfe_price=position.max_favorable_price,
            mae_price=position.max_adverse_price,
            mfe_pnl=position.max_favorable_pnl,
            mae_pnl=position.max_adverse_pnl,
            mfe_r=mfe_r,
            mae_r=mae_r,
            mfe_reached_at=position.mfe_reached_at,
            mae_reached_at=position.mae_reached_at
        )
        
        self.trades.append(trade)

        # Track realized P&L after costs for daily loss caps
        day_key = exit_time.normalize()
        self._realized_pnl_after_costs_per_day[day_key] = (
            self._realized_pnl_after_costs_per_day.get(day_key, 0.0) + float(pnl_after_costs)
        )
        
        # MONTE CARLO COMPATIBLE: Track equity and returns
        self.equity_list.append(equity_after)
        self.trade_returns_list.append(trade_return)
    
    def _close_all_positions(
        self,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str
    ) -> None:
        """Close all open positions.
        
        Args:
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            self._close_position(position, exit_time, exit_price, exit_reason)
        self.positions.clear()
    
    def _check_margin_calls(
        self,
        current_time: pd.Timestamp,
        current_price: float
    ) -> None:
        """Check for margin calls and force liquidation if needed.
        
        For Binance-style futures: Uses maintenance margin (equity < maintenance_margin)
        For traditional futures/other: Uses margin_call_level (equity < margin_used * margin_call_level)
        
        Args:
            current_time: Current timestamp
            current_price: Current market price
        """
        # Update unrealized P&L
        self.account.update_unrealized_pnl(self.broker, current_price)
        
        # Calculate total maintenance margin for all positions (if using Binance-style futures)
        total_maintenance_margin = None
        if self.market_spec.maintenance_margin_rate is not None:
            # Binance-style futures: Calculate maintenance margin for all positions
            total_maintenance_margin = 0.0
            for position in self.account.open_positions:
                position_maintenance = self.broker.calculate_maintenance_margin_required(
                    position.entry_price,
                    position.size,
                    current_price
                )
                total_maintenance_margin += position_maintenance
        
        # Check for margin call (passes maintenance_margin if available)
        if self.broker.check_margin_call(
            self.account.equity,
            self.account.used_margin,
            maintenance_margin=total_maintenance_margin
        ):
            # Force close all positions
            self._close_all_positions(current_time, current_price, 'margin_call')

    def _should_stop_signal_search(self, current_time: pd.Timestamp) -> bool:
        """Check if signal search should be stopped at this time.
        
        This method checks if the current time is past the stop signal search time,
        which is configured as minutes before the flatten time.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if signal search should be stopped, False otherwise
        """
        # If stop_signal_search_minutes_before is not configured, never stop
        if self._stop_signal_search_minutes is None:
            return False
        
        # Calculate current time in minutes (0-1439 for 24 hours)
        current_minutes = current_time.hour * 60 + current_time.minute
        
        # Check if current time is past the stop signal search time
        return current_minutes >= self._stop_signal_search_minutes
    
    def _check_flatten_time(
        self,
        current_time: pd.Timestamp,
        current_price: float
    ) -> None:
        """Flatten all positions at configured end-of-day time."""
        if not self.flatten_enabled or self._flatten_minutes is None:
            return
        
        # Ensure once per day
        current_date = current_time.normalize()
        if self._flatten_closed_date is not None and self._flatten_closed_date == current_date:
            return
        
        minutes = current_time.hour * 60 + current_time.minute
        if minutes >= self._flatten_minutes:
            self._close_all_positions(current_time, current_price, 'flatten_time')
            self._flatten_closed_date = current_date
    
    def _update_account_state(
        self,
        current_time: pd.Timestamp,
        current_price: float
    ) -> None:
        """Update account state and record equity.
        
        CRITICAL: equity_curve (bar-by-bar) is derived from account.equity which includes
        unrealized P&L. equity_list (trade-based) is the single source of truth for MC.
        Both are maintained for different purposes but should align at trade boundaries.
        
        Args:
            current_time: Current timestamp
            current_price: Current market price
        """
        # Update unrealized P&L for all positions
        self.account.update_unrealized_pnl(self.broker, current_price)
        
        # Update MFE/MAE for all open positions
        self._update_mfe_mae(current_price, current_time)
        
        # Record equity (bar-by-bar for reporting)
        self.account.record_equity(current_time)
        
        # CRITICAL FIX: Ensure equity_curve aligns with equity_list at trade boundaries
        # If we just closed a trade, use the equity from equity_list
        if len(self.equity_list) > 0:
            # Use trade-based equity as source of truth, but account.equity includes unrealized P&L
            # For bar-by-bar reporting, use account.equity (includes unrealized)
            # For MC compatibility, use equity_list (trade-based only)
            self.equity_curve.append(self.account.equity)
        else:
            self.equity_curve.append(self.account.equity)
        
        self.equity_timestamps.append(current_time)
        
        # Record margin utilization
        if self.account.used_margin > 0:
            margin_util = self.account.used_margin / self.account.equity if self.account.equity > 0 else 0.0
        else:
            margin_util = 0.0
        self.margin_utilization.append(margin_util)
    
    def _update_mfe_mae(self, current_price: float, current_time: Optional[pd.Timestamp] = None) -> None:
        """Update Maximum Favorable/Adverse Excursion for all open positions.
        
        MFE/MAE is calculated based on the initial position size to track
        the maximum favorable/adverse excursion that was ever reached during
        the trade's lifetime, regardless of partial exits.
        
        Args:
            current_price: Current market price
            current_time: Current timestamp (for tracking when MFE/MAE was reached)
        """
        for position in self.positions:
            # Calculate current unrealized P&L based on INITIAL position size
            # This ensures MFE/MAE reflects the maximum that was ever reached,
            # even if partial exits reduced the position size
            if position.direction == 'long':
                current_pnl = (current_price - position.entry_price) * position.initial_size
            else:  # short
                current_pnl = (position.entry_price - current_price) * position.initial_size
            
            # Update MFE (Maximum Favorable Excursion)
            if current_pnl > position.max_favorable_pnl:
                position.max_favorable_pnl = current_pnl
                position.max_favorable_price = current_price
                if current_time is not None:
                    position.mfe_reached_at = current_time
            
            # Update MAE (Maximum Adverse Excursion - industry standard terminology)
            if current_pnl < position.max_adverse_pnl:
                position.max_adverse_pnl = current_pnl
                position.max_adverse_price = current_price
                if current_time is not None:
                    position.mae_reached_at = current_time
    
    def _validate_accounting(self, current_price: float) -> None:
        """Validate accounting invariants.
        
        Args:
            current_price: Current market price for validation
        """
        # Update unrealized P&L
        self.account.update_unrealized_pnl(self.broker, current_price)
        
        # Validate invariant: equity == cash + unrealized_pnl
        if not self.account.validate_invariant(self.broker, current_price):
            expected_equity = self.account.cash + self.account.unrealized_pnl
            actual_equity = self.account.equity
            raise ValueError(
                f"Accounting invariant violated: "
                f"cash={self.account.cash:.2f}, "
                f"unrealized_pnl={self.account.unrealized_pnl:.2f}, "
                f"expected_equity={expected_equity:.2f}, "
                f"actual_equity={actual_equity:.2f}"
            )
        
        # Validate final equity matches trade P&L (using pnl_after_costs)
        total_net_pnl = sum(trade.pnl_after_costs for trade in self.trades)
        expected_final_equity = self.initial_capital + total_net_pnl
        actual_final_equity = self.account.equity
        
        # Allow small floating point errors
        if abs(expected_final_equity - actual_final_equity) > 1e-3:
            warnings.warn(
                f"Final equity mismatch: "
                f"expected={expected_final_equity:.2f} "
                f"(initial={self.initial_capital:.2f} + net_pnl={total_net_pnl:.2f}), "
                f"actual={actual_final_equity:.2f}. "
                f"Difference: {abs(expected_final_equity - actual_final_equity):.2f}"
            )
    
    def _create_result(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> BacktestResult:
        # Log cumulative rejection summary
        if hasattr(self, '_cumulative_rejections'):
            total_rejections = sum(self._cumulative_rejections.values())
            if total_rejections > 0:
                self.logger.debug(f"Cumulative Signal Rejections Summary:")
                self.logger.debug(f"  Total rejections: {total_rejections}")
                for key, value in self._cumulative_rejections.items():
                    if value > 0:
                        self.logger.debug(f"  {key}: {value}")
                self.logger.debug(f"  Pending signals at end: {len(self.pending_signals)}")
        
        # Log immediate execution failures (master filters OFF)
        if hasattr(self, '_immediate_exec_failures'):
            total_failures = sum(self._immediate_exec_failures.values())
            if total_failures > 0:
                self.logger.debug(f"Immediate Execution Failures (Master Filters OFF):")
                self.logger.debug(f"  Total failures: {total_failures}")
                for key, value in self._immediate_exec_failures.items():
                    if value > 0:
                        self.logger.debug(f"  {key}: {value}")
        
        # Log daily limit rejections
        if hasattr(self, '_daily_limit_rejections') and self._daily_limit_rejections > 0:
            self.logger.debug(f"Daily Trade Limit Rejections: {self._daily_limit_rejections}")
        
        """Create BacktestResult from backtest execution - Monte Carlo compatible.
        
        Args:
            start_time: Backtest start time
            end_time: Backtest end time
            
        Returns:
            BacktestResult object with integer-indexed equity_curve and trade_returns
        """
        # MONTE CARLO COMPATIBLE: Build integer-indexed equity curve
        # equity_curve[0] = initial_capital
        # equity_curve[i] = equity after trade i
        if len(self.equity_list) == 0:
            # No trades - just initial capital
            equity_list = [self.initial_capital]
        else:
            equity_list = self.equity_list
        
        # Ensure equity_curve has length N+1 (initial + after each trade)
        if len(equity_list) != len(self.trades) + 1:
            # Pad or truncate to match
            if len(equity_list) < len(self.trades) + 1:
                # Pad with last value
                last_equity = equity_list[-1] if equity_list else self.initial_capital
                equity_list.extend([last_equity] * (len(self.trades) + 1 - len(equity_list)))
            else:
                # Truncate to N+1
                equity_list = equity_list[:len(self.trades) + 1]
        
        # Create integer-indexed equity curve (0..N)
        equity_curve = pd.Series(equity_list, index=range(len(equity_list)))
        
        # MONTE CARLO COMPATIBLE: Build trade_returns array
        # Ensure trade_returns matches length of trades
        if len(self.trade_returns_list) != len(self.trades):
            # Recompute if mismatch (shouldn't happen, but safety check)
            trade_returns = []
            for i, trade in enumerate(self.trades):
                equity_before = equity_curve.iloc[i] if i < len(equity_curve) else self.initial_capital
                if equity_before > 0:
                    trade_return = trade.pnl_after_costs / equity_before
                else:
                    trade_return = 0.0
                trade_returns.append(trade_return)
            trade_returns = np.array(trade_returns, dtype=float)
        else:
            trade_returns = np.array(self.trade_returns_list, dtype=float)
        
        # Calculate final capital from equity curve
        final_capital = float(equity_curve.iloc[-1])
        total_pnl = final_capital - self.initial_capital
        
        # Calculate basic metrics using pnl_after_costs
        winning_trades = [t for t in self.trades if t.pnl_after_costs > 0]
        losing_trades = [t for t in self.trades if t.pnl_after_costs <= 0]
        
        win_rate = (len(winning_trades) / len(self.trades) * 100.0) if self.trades else 0.0
        
        total_commission = sum(trade.commission for trade in self.trades)
        total_slippage = sum(trade.slippage for trade in self.trades)
        
        # Calculate max drawdown from equity_curve
        if len(equity_curve) > 1:
            peak = equity_curve.expanding().max()
            drawdown_pct = ((equity_curve - peak) / peak) * 100.0
            max_drawdown = abs(float(drawdown_pct.min()))
            max_drawdown = min(max_drawdown, 100.0)  # Cap at 100%
        else:
            max_drawdown = 0.0
        
        # Calculate exposure and leverage stats (for backward compatibility)
        exposure_stats = self._calculate_exposure_stats()
        leverage_stats = self._calculate_leverage_stats()
        
        # Create timestamped equity series for backward compatibility
        if len(self.equity_curve) == len(self.equity_timestamps) and len(self.equity_curve) > 0:
            equity_series_timestamped = pd.Series(self.equity_curve, index=self.equity_timestamps)
        else:
            # Fallback to integer-indexed equity curve if timestamped version not available
            equity_series_timestamped = equity_curve
        
        # Create drawdown curve for backward compatibility
        if len(equity_series_timestamped) > 1:
            peak_ts = equity_series_timestamped.expanding().max()
            drawdown_ts = ((equity_series_timestamped - peak_ts) / peak_ts) * 100.0
        elif len(equity_series_timestamped) == 1:
            # Single point - no drawdown
            drawdown_ts = pd.Series([0.0], index=equity_series_timestamped.index)
        else:
            # Empty - create empty series with same index type
            drawdown_ts = pd.Series([], dtype=float)
        
        # Create margin utilization series (for backward compatibility)
        if len(self.margin_utilization) == len(self.equity_timestamps) and len(self.equity_timestamps) > 0:
            margin_util_series = pd.Series(self.margin_utilization, index=self.equity_timestamps)
        else:
            margin_util_series = None
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            equity_curve=equity_curve,  # Integer-indexed for MC compatibility
            trade_returns=trade_returns,  # Precomputed trade returns
            trades=self.trades,
            total_pnl=total_pnl,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            total_commission=total_commission,
            total_slippage=total_slippage,
            max_drawdown=max_drawdown,
            strategy_name=self.strategy.name,
            symbol=self.market_spec.symbol,
            # Backward compatibility fields
            win_rate=win_rate,
            drawdown_curve=drawdown_ts,
            exposure_stats=exposure_stats,
            leverage_stats=leverage_stats,
            margin_utilization=margin_util_series
            ,
            entry_tf=(self._result_entry_tf or ""),
            price_df=self._result_price_df,
        )
    
    def _calculate_exposure_stats(self) -> Dict:
        """Calculate exposure statistics.
        
        Returns:
            Dictionary with exposure statistics
        """
        if not self.trades:
            return {}
        
        # Calculate average position size
        avg_size = np.mean([t.size for t in self.trades])
        max_size = max([t.size for t in self.trades])
        min_size = min([t.size for t in self.trades])
        
        # Calculate average notional value
        avg_notional = np.mean([t.entry_price * t.size for t in self.trades])
        max_notional = max([t.entry_price * t.size for t in self.trades])
        
        return {
            'avg_position_size': avg_size,
            'max_position_size': max_size,
            'min_position_size': min_size,
            'avg_notional_value': avg_notional,
            'max_notional_value': max_notional
        }
    
    def _calculate_leverage_stats(self) -> Dict:
        """Calculate leverage usage statistics.
        
        Returns:
            Dictionary with leverage statistics
        """
        if not self.trades:
            return {}
        
        # Calculate average leverage used
        leverage_ratios = []
        for trade in self.trades:
            notional = trade.entry_price * trade.size
            margin = notional / self.market_spec.leverage
            leverage_ratio = notional / margin if margin > 0 else 0.0
            leverage_ratios.append(leverage_ratio)
        
        avg_leverage = np.mean(leverage_ratios) if leverage_ratios else 0.0
        max_leverage = max(leverage_ratios) if leverage_ratios else 0.0
        
        return {
            'avg_leverage': avg_leverage,
            'max_leverage': max_leverage,
            'market_leverage': self.market_spec.leverage
        }

