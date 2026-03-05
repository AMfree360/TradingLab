"""Live trading engine that executes strategies in real-time."""

import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import pandas as pd
import numpy as np

from strategies.base import StrategyBase
from adapters.execution.binance_futures import BinanceFuturesAdapter, Position as AdapterPosition
from engine.market import MarketSpec
from engine.account import AccountState
from engine.broker import BrokerModel

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """Engine for live trading with real-time execution."""
    
    def __init__(
        self,
        strategy: StrategyBase,
        adapter: BinanceFuturesAdapter,
        market_spec: MarketSpec,
        market_specs_by_symbol: Optional[Dict[str, MarketSpec]] = None,
        initial_capital: float = 10000.0,
        max_daily_loss_pct: float = 5.0,
        max_drawdown_pct: float = 15.0
    ):
        """
        Initialize live trading engine.
        
        Args:
            strategy: Strategy instance
            adapter: Exchange API adapter
            market_spec: Market specification
            initial_capital: Starting capital
            max_daily_loss_pct: Maximum daily loss percentage before stopping
            max_drawdown_pct: Maximum drawdown percentage before stopping
        """
        self.strategy = strategy
        self.adapter = adapter
        self.market_spec = market_spec
        self.market_specs_by_symbol: Dict[str, MarketSpec] = dict(market_specs_by_symbol or {})
        if self.market_spec is not None and getattr(self.market_spec, 'symbol', None):
            self.market_specs_by_symbol.setdefault(self.market_spec.symbol, self.market_spec)
        self.initial_capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        # Initialize account and broker
        self.account = AccountState(initial_capital)
        self.broker = BrokerModel(market_spec)
        
        # Trading state
        self.symbol = market_spec.symbol
        self.active_positions: Dict[str, AdapterPosition] = {}
        self.active_stop_orders: Dict[str, str] = {}  # position_id -> stop_order_id
        self.active_stop_prices: Dict[str, float] = {}  # symbol -> stop_price
        self.daily_pnl = 0.0
        self.daily_start_balance = initial_capital
        self.peak_equity = initial_capital

        # Optional end-of-day flattening (parity with backtest execution config)
        exec_cfg = getattr(strategy.config, 'execution', None)
        self.flatten_enabled = getattr(exec_cfg, 'flatten_enabled', True) if exec_cfg else True
        self.flatten_time_str = getattr(exec_cfg, 'flatten_time', '21:30') if self.flatten_enabled else None
        self.stop_signal_search_minutes_before = getattr(exec_cfg, 'stop_signal_search_minutes_before', None) if exec_cfg else None
        self._flatten_minutes: Optional[int] = None
        self._stop_signal_search_minutes: Optional[int] = None
        self._flatten_closed_date: Optional[pd.Timestamp] = None

        if self.flatten_enabled and self.flatten_time_str:
            try:
                hours, mins = str(self.flatten_time_str).split(":")
                self._flatten_minutes = int(hours) * 60 + int(mins)

                if self.stop_signal_search_minutes_before is not None:
                    self._stop_signal_search_minutes = self._flatten_minutes - int(self.stop_signal_search_minutes_before)
                    if self._stop_signal_search_minutes < 0:
                        self._stop_signal_search_minutes += 24 * 60
            except Exception:
                self._flatten_minutes = None
                self._stop_signal_search_minutes = None

        # Track realized P&L after costs per day for config-driven daily loss caps.
        self._realized_pnl_after_costs_per_day: Dict[pd.Timestamp, float] = {}

        # Track day-start equity baseline (used when sizing_mode != account_size).
        self._day_start_equity_by_day: Dict[pd.Timestamp, float] = {}
        self._current_day_key: Optional[pd.Timestamp] = None

        # Track entry details for deterministic realized P&L in CLOSE signals.
        self._entry_side_by_symbol: Dict[str, str] = {}
        self._entry_price_by_symbol: Dict[str, float] = {}
        self._entry_qty_by_symbol: Dict[str, float] = {}

        # Track entries per day for parity with backtest trade limits.
        # Keyed by UTC-normalized day.
        self._entries_per_day: Dict[pd.Timestamp, int] = {}

        # Market data cache (optional). Used for ATR stop fallback.
        self.recent_bars: Optional[pd.DataFrame] = None
        
        # Statistics
        self.trades_executed = 0
        self.trades_skipped = 0
        
    def update_positions(self):
        """Update active positions from exchange."""
        # Fetch all open positions so portfolio risk cap can be enforced across symbols.
        positions = self.adapter.get_positions(None)
        self.active_positions = {pos.symbol: pos for pos in positions if abs(pos.size) > 0.0001}

        # Clean up stop tracking for symbols that are no longer open
        open_symbols = set(self.active_positions.keys())
        for sym in list(self.active_stop_orders.keys()):
            if sym not in open_symbols:
                self.active_stop_orders.pop(sym, None)
        for sym in list(self.active_stop_prices.keys()):
            if sym not in open_symbols:
                self.active_stop_prices.pop(sym, None)
        
        # Update account with unrealized P&L
        total_unrealized = sum(pos.unrealized_pnl for pos in self.active_positions.values())
        # Note: In live trading, we track this separately from account equity

    def _get_market_spec_for_symbol(self, symbol: str) -> MarketSpec:
        spec = self.market_specs_by_symbol.get(symbol)
        if spec is not None:
            return spec
        return self.market_spec
        
    def check_safety_limits(self) -> Tuple[bool, Optional[str]]:
        """Check if safety limits are exceeded.
        
        Returns:
            (should_stop, reason)
        """
        balance = self.adapter.get_account_balance()
        current_equity = balance.total_balance
        
        # Check daily loss (config override if provided)
        trade_limits_cfg = getattr(self.strategy.config, 'trade_limits', None)
        effective_daily_loss_pct = getattr(trade_limits_cfg, 'max_daily_loss_pct', None) if trade_limits_cfg else None
        if effective_daily_loss_pct is None:
            effective_daily_loss_pct = self.max_daily_loss_pct

        if effective_daily_loss_pct is not None:
            try:
                daily_loss_pct = ((self.daily_start_balance - current_equity) / self.daily_start_balance) * 100
                if daily_loss_pct >= float(effective_daily_loss_pct):
                    return True, f"Daily loss limit exceeded: {daily_loss_pct:.2f}%"
            except Exception:
                pass
        
        # Check drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        drawdown_pct = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        if drawdown_pct >= self.max_drawdown_pct:
            return True, f"Drawdown limit exceeded: {drawdown_pct:.2f}%"
        
        return False, None
    
    def process_signal(self, signal: Dict[str, Any], current_price: float, current_time: pd.Timestamp):
        """Process a trading signal from the strategy.
        
        Args:
            signal: Signal dict with 'action' ('BUY', 'SELL', 'CLOSE'), 'quantity', etc.
            current_price: Current market price
            current_time: Current timestamp
        """
        action = signal.get('action')
        if not action:
            return
        
        # Check safety limits
        should_stop, reason = self.check_safety_limits()
        if should_stop:
            logger.warning(f"Trading stopped: {reason}")
            return
        
        # Update positions
        self.update_positions()

        # Optional EOD flattening (safe no-op if disabled / unconfigured)
        try:
            self._check_flatten_time(current_time=current_time, current_price=float(current_price))
        except Exception:
            pass
        
        # Handle different actions
        if action == 'CLOSE':
            self._close_all_positions(current_price=current_price, current_time=current_time)
        elif action in ['BUY', 'SELL']:
            self._process_entry_signal(signal, current_price, current_time)
        elif action == 'UPDATE_STOP':
            self._update_stop_loss(signal, current_price)

    @staticmethod
    def _day_key(ts: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(ts).normalize()

    def _ensure_day_initialized(self, ts: pd.Timestamp) -> None:
        day_key = self._day_key(ts)
        self._current_day_key = day_key
        if day_key not in self._day_start_equity_by_day:
            try:
                bal = self.adapter.get_account_balance()
                self._day_start_equity_by_day[day_key] = float(getattr(bal, 'total_balance', self.initial_capital))
            except Exception:
                self._day_start_equity_by_day[day_key] = float(self.initial_capital)

    def _get_day_base_equity_for_limits(self, ts: pd.Timestamp) -> float:
        risk_cfg = getattr(self.strategy.config, 'risk', None)
        sizing_mode = getattr(risk_cfg, 'sizing_mode', 'account_size') if risk_cfg else 'account_size'
        if sizing_mode == 'account_size':
            acct_size = getattr(risk_cfg, 'account_size', None) if risk_cfg else None
            return float(acct_size) if acct_size is not None else float(self.initial_capital)

        day_key = self._day_key(ts)
        base = self._day_start_equity_by_day.get(day_key)
        if base is None:
            self._ensure_day_initialized(ts)
            base = self._day_start_equity_by_day.get(day_key, float(self.initial_capital))
        return float(base)

    def _daily_loss_limit_reached(self, ts: pd.Timestamp) -> bool:
        trade_limits_cfg = getattr(self.strategy.config, 'trade_limits', None)
        max_daily_loss_pct = getattr(trade_limits_cfg, 'max_daily_loss_pct', None) if trade_limits_cfg else None
        if max_daily_loss_pct is None:
            return False
        try:
            pct = float(max_daily_loss_pct)
        except (TypeError, ValueError):
            return False
        if pct <= 0:
            return False

        day_key = self._day_key(ts)
        realized = float(self._realized_pnl_after_costs_per_day.get(day_key, 0.0))
        base = self._get_day_base_equity_for_limits(ts)
        limit_amount = base * (pct / 100.0)
        return realized <= -limit_amount

    def _should_stop_signal_search(self, current_time: pd.Timestamp) -> bool:
        """Match BacktestEngine semantics: stop searching when minutes >= configured stop time."""
        if self._stop_signal_search_minutes is None:
            return False
        t = pd.Timestamp(current_time)
        minutes = int(t.hour) * 60 + int(t.minute)
        return minutes >= int(self._stop_signal_search_minutes)

    def _check_flatten_time(self, *, current_time: pd.Timestamp, current_price: float) -> None:
        """Flatten all open positions at configured end-of-day time (best-effort)."""
        if not getattr(self, 'flatten_enabled', False) or self._flatten_minutes is None:
            return

        t = pd.Timestamp(current_time)
        current_date = t.normalize()
        if self._flatten_closed_date is not None and self._flatten_closed_date == current_date:
            return

        minutes = int(t.hour) * 60 + int(t.minute)
        if minutes >= int(self._flatten_minutes):
            self._close_all_positions(current_price=float(current_price), current_time=t)
            self._flatten_closed_date = current_date

    def update_recent_bars(self, bars: pd.DataFrame) -> None:
        """Update cached bars for ATR stop fallback.

        Expected columns: open, high, low, close, volume (at minimum high/low/close).
        Index should be time-ordered.
        """
        if bars is None or len(bars) == 0:
            return
        self.recent_bars = bars.copy()
    
    def _process_entry_signal(
        self,
        signal: Dict[str, Any],
        current_price: float,
        current_time: pd.Timestamp
    ):
        """Process an entry signal."""
        action = signal.get('action')  # 'BUY' or 'SELL'
        stop_price = signal.get('stop_price')
        target_price = signal.get('target_price')

        quantity = signal.get('quantity')
        
        # Check if we already have a position
        if self.symbol in self.active_positions:
            logger.info(f"Position already exists for {self.symbol}, skipping entry")
            self.trades_skipped += 1
            return

        # Apply deterministic time-based trade gating (session/blackout windows, etc).
        if not self._is_trade_allowed_at(current_time):
            logger.info("Trade filtered out by trade_filters; skipping entry")
            self.trades_skipped += 1
            return

        # Stop searching for new entries X minutes before flatten time (if configured).
        if self._should_stop_signal_search(current_time):
            logger.info("Stop-signal-search window active; skipping entry")
            self.trades_skipped += 1
            return

        # Initialize per-day tracking (safe even if called out-of-loop)
        self._ensure_day_initialized(current_time)

        # Enforce max daily loss cap (backtest parity): stop taking new trades for rest of day.
        if self._daily_loss_limit_reached(current_time):
            logger.info("Max daily loss reached; skipping entry")
            self.trades_skipped += 1
            return

        # Enforce max trades per day (strategy-config parity with backtest).
        try:
            trade_limits_cfg = getattr(self.strategy.config, 'trade_limits', None)
            max_trades_per_day = getattr(trade_limits_cfg, 'max_trades_per_day', None) if trade_limits_cfg else None
            if max_trades_per_day is not None:
                day_key = pd.Timestamp(current_time).normalize()
                taken = int(self._entries_per_day.get(day_key, 0))
                if taken >= int(max_trades_per_day):
                    logger.info("Max trades per day reached; skipping entry")
                    self.trades_skipped += 1
                    return
        except Exception:
            pass

        # Validate/compute stop price (portfolio stop fallback)
        entry_side = 'BUY' if action == 'BUY' else 'SELL'
        dir_int = 1 if entry_side == 'BUY' else -1
        stop_price_final = self._get_stop_price_with_fallback_live(
            stop_price=stop_price,
            reference_price=float(current_price),
            dir_int=dir_int,
            recent_bars=self.recent_bars,
        )
        if stop_price_final is None:
            logger.warning("Missing/invalid stop_price and no fallback available; skipping entry")
            self.trades_skipped += 1
            return

        # If quantity wasn't provided by the strategy/signal, compute it from risk config
        # to match BacktestEngine sizing semantics.
        if quantity is None or float(quantity) <= 0:
            try:
                risk_cfg = getattr(self.strategy.config, 'risk', None)
                risk_pct = getattr(risk_cfg, 'risk_per_trade_pct', None) if risk_cfg else None
                if risk_pct is None:
                    logger.warning(f"Missing quantity and risk_per_trade_pct; cannot size entry")
                    self.trades_skipped += 1
                    return

                quantity = self._calculate_position_size_live(
                    entry_price=float(current_price),
                    stop_price=float(stop_price_final),
                    risk_pct=float(risk_pct),
                    current_time=pd.Timestamp(current_time),
                )
            except Exception as e:
                logger.warning(f"Failed to auto-size quantity: {e}")
                self.trades_skipped += 1
                return

        if quantity is None or float(quantity) <= 0:
            logger.warning(f"Invalid quantity in signal after sizing: {quantity}")
            self.trades_skipped += 1
            return

        # Enforce portfolio open-risk cap by scaling down quantity if needed
        quantity = self._apply_portfolio_open_risk_cap(
            symbol=self.symbol,
            entry_price=float(current_price),
            stop_price=float(stop_price_final),
            quantity=float(quantity),
        )
        if quantity is None or quantity <= 0:
            logger.info("Portfolio open-risk cap prevents new entry; skipping")
            self.trades_skipped += 1
            return

        # Per-symbol quantity normalization & minimum size enforcement.
        sym_spec = self._get_market_spec_for_symbol(self.symbol)
        try:
            q_prec = int(getattr(sym_spec, 'quantity_precision', 2))
            quantity = round(float(quantity), q_prec)
        except Exception:
            quantity = float(quantity)

        min_size = float(getattr(sym_spec, 'min_trade_size', 0.0) or 0.0)
        if min_size > 0 and float(quantity) < min_size:
            logger.info("Quantity below min_trade_size; skipping entry")
            self.trades_skipped += 1
            return
        
        # Check balance
        balance = self.adapter.get_account_balance()
        if balance.available_balance <= 0:
            logger.warning("Insufficient balance for trade")
            self.trades_skipped += 1
            return
        
        # Calculate margin required
        margin_required = sym_spec.calculate_margin(current_price, quantity)
        if margin_required > balance.available_balance:
            logger.warning(f"Insufficient margin: required {margin_required}, available {balance.available_balance}")
            self.trades_skipped += 1
            return
        
        # Place entry order
        try:
            side = 'BUY' if action == 'BUY' else 'SELL'
            order = self.adapter.place_market_order(
                symbol=self.symbol,
                side=side,
                quantity=quantity
            )
            
            logger.info(f"Entry order placed: {side} {quantity} {self.symbol} @ {current_price}")
            self.trades_executed += 1

            # Record entry details for deterministic realized P&L calculations on CLOSE.
            try:
                self._entry_side_by_symbol[self.symbol] = str(side)
                self._entry_price_by_symbol[self.symbol] = float(current_price)
                self._entry_qty_by_symbol[self.symbol] = float(quantity)
            except Exception:
                pass

            try:
                day_key = pd.Timestamp(current_time).normalize()
                self._entries_per_day[day_key] = int(self._entries_per_day.get(day_key, 0)) + 1
            except Exception:
                pass
            
            # Place stop loss if provided
            if stop_price_final is not None:
                self._place_stop_loss(float(quantity), float(stop_price_final), side)
            
        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            self.trades_skipped += 1

    def _calculate_position_size_live(
        self,
        *,
        entry_price: float,
        stop_price: float,
        risk_pct: float,
        current_time: Optional[pd.Timestamp] = None,
    ) -> float:
        """Backtest-style position sizing for live execution.

        Uses strategy.config.risk.sizing_mode and risk_per_trade_pct.
        - account_size: deterministic sizing from configured account_size/initial_capital
        - daily_equity/equity: uses adapter balance (non-deterministic across live)
        """
        if entry_price <= 0 or stop_price <= 0:
            return 0.0

        price_risk = abs(float(entry_price) - float(stop_price))
        if price_risk <= 0:
            return 0.0

        risk_cfg = getattr(self.strategy.config, 'risk', None)
        sizing_mode = getattr(risk_cfg, 'sizing_mode', 'account_size') if risk_cfg else 'account_size'
        acct_size = getattr(risk_cfg, 'account_size', None) if risk_cfg else None

        if sizing_mode == 'account_size':
            base = float(acct_size) if acct_size is not None else float(self.initial_capital)
        elif sizing_mode == 'daily_equity':
            ts = pd.Timestamp(current_time) if current_time is not None else pd.Timestamp.utcnow()
            self._ensure_day_initialized(ts)
            base = float(self._day_start_equity_by_day.get(self._day_key(ts), float(self.initial_capital)))
        else:
            try:
                bal = self.adapter.get_account_balance()
                base = float(getattr(bal, 'total_balance', self.initial_capital))
            except Exception:
                base = float(self.initial_capital)

        if base <= 0:
            return 0.0

        risk_amount = float(base) * (float(risk_pct) / 100.0)
        if risk_amount <= 0:
            return 0.0

        # Asset-class specific sizing
        sym_spec = self._get_market_spec_for_symbol(self.symbol)

        if getattr(sym_spec, 'asset_class', None) == 'forex' and getattr(sym_spec, 'contract_size', None) is not None:
            pip_size = float(getattr(sym_spec, 'pip_value', 0.0001) or 0.0001)
            stop_loss_pips = float(price_risk) / float(pip_size)
            if stop_loss_pips < 1.0:
                return 0.0
            pip_value_per_unit = float(sym_spec.calculate_pip_value_per_unit(entry_price))
            if pip_value_per_unit <= 0:
                return 0.0
            quantity = float(sym_spec.calculate_quantity_from_risk(risk_amount, stop_loss_pips, entry_price))
            if quantity <= 0:
                return 0.0
            lot_size = quantity / float(sym_spec.contract_size)
            if lot_size < float(getattr(sym_spec, 'min_trade_size', 0.0)):
                return 0.0
        elif getattr(sym_spec, 'asset_class', None) == 'futures' and getattr(sym_spec, 'contract_size', None) is not None:
            dollar_risk_per_contract = float(price_risk) * float(sym_spec.contract_size)
            if dollar_risk_per_contract <= 0:
                return 0.0
            quantity = float(risk_amount) / float(dollar_risk_per_contract)

            # Match BacktestEngine behavior:
            # - Portfolio cap ON: floor to whole contracts, require >= 1
            # - Portfolio cap OFF: allow fractional contracts, but minimum 1
            portfolio_cfg = getattr(self.strategy.config, 'portfolio', None)
            portfolio_enabled = bool(getattr(portfolio_cfg, 'enabled', False)) if portfolio_cfg else False
            max_open_risk_pct = getattr(portfolio_cfg, 'max_open_risk_pct', None) if portfolio_cfg else None
            if portfolio_enabled and max_open_risk_pct is not None and float(max_open_risk_pct) > 0:
                quantity = float(np.floor(quantity))
                quantity = max(1.0, quantity)
            else:
                quantity = max(1.0, float(quantity))
        else:
            quantity = float(risk_amount) / float(price_risk)

        if quantity <= 0:
            return 0.0

        # Respect exchange precision/min size
        qty_prec = int(getattr(sym_spec, 'quantity_precision', 2))
        quantity = round(float(quantity), qty_prec)
        if quantity < float(getattr(sym_spec, 'min_trade_size', 0.0)):
            return 0.0
        return float(quantity)

    def _is_trade_allowed_at(self, ts: pd.Timestamp) -> bool:
        """Deterministic, timestamp-only check whether trade filters allow an entry at ts.

        Mirrors BacktestEngine behavior to reduce backtest/live mismatches.
        """
        try:
            from gui_launcher.app import compute_trade_filters_mask
            import pandas as pd

            idx = pd.DatetimeIndex([pd.Timestamp(ts)])
            rules = getattr(self.strategy.config, 'trade_filters', []) or []
            mask = compute_trade_filters_mask(idx, rules, default_tz=None)
            return bool(mask.iloc[0])
        except Exception:
            # Be permissive on errors to avoid silently blocking live trading.
            return True
    
    def _place_stop_loss(self, quantity: float, stop_price: float, entry_side: str):
        """Place a stop loss order."""
        try:
            # Stop order side is opposite of entry
            stop_side = 'SELL' if entry_side == 'BUY' else 'BUY'

            sym_spec = self._get_market_spec_for_symbol(self.symbol)
            try:
                q_prec = int(getattr(sym_spec, 'quantity_precision', 2))
                quantity = round(float(quantity), q_prec)
            except Exception:
                quantity = float(quantity)

            min_size = float(getattr(sym_spec, 'min_trade_size', 0.0) or 0.0)
            if min_size > 0 and float(quantity) < min_size:
                logger.info("Stop quantity below min_trade_size; skipping stop placement")
                return
            
            stop_order = self.adapter.place_stop_market_order(
                symbol=self.symbol,
                side=stop_side,
                quantity=quantity,
                stop_price=stop_price,
                reduce_only=True
            )
            
            self.active_stop_orders[self.symbol] = stop_order.order_id
            self.active_stop_prices[self.symbol] = float(stop_price)
            logger.info(f"Stop loss placed: {stop_side} {quantity} {self.symbol} @ {stop_price}")
            
        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")
    
    def _update_stop_loss(self, signal: Dict[str, Any], current_price: float):
        """Update stop loss (trailing stop)."""
        if self.symbol not in self.active_positions:
            return
        
        new_stop_price = signal.get('stop_price')
        if not new_stop_price:
            return
        
        # Cancel old stop order
        if self.symbol in self.active_stop_orders:
            try:
                self.adapter.cancel_order(self.symbol, self.active_stop_orders[self.symbol])
            except Exception as e:
                logger.warning(f"Failed to cancel old stop order: {e}")
        
        # Place new stop order
        position = self.active_positions[self.symbol]
        stop_side = 'SELL' if position.side == 'LONG' else 'BUY'
        
        try:
            sym_spec = self._get_market_spec_for_symbol(self.symbol)
            qty = float(position.size)
            try:
                q_prec = int(getattr(sym_spec, 'quantity_precision', 2))
                qty = round(float(qty), q_prec)
            except Exception:
                qty = float(qty)

            min_size = float(getattr(sym_spec, 'min_trade_size', 0.0) or 0.0)
            if min_size > 0 and float(qty) < min_size:
                logger.info("Stop update quantity below min_trade_size; skipping stop update")
                return

            stop_order = self.adapter.place_stop_market_order(
                symbol=self.symbol,
                side=stop_side,
                quantity=qty,
                stop_price=new_stop_price,
                reduce_only=True
            )
            
            self.active_stop_orders[self.symbol] = stop_order.order_id
            self.active_stop_prices[self.symbol] = float(new_stop_price)
            logger.info(f"Stop loss updated: {stop_side} {position.size} {self.symbol} @ {new_stop_price}")
            
        except Exception as e:
            logger.error(f"Failed to update stop loss: {e}")
    
    def _close_all_positions(self, *, current_price: Optional[float] = None, current_time: Optional[pd.Timestamp] = None):
        """Close all open positions.

        If current_price/current_time are provided, updates realized P&L tracking
        for max_daily_loss_pct gating.
        """
        if current_time is not None:
            try:
                self._ensure_day_initialized(current_time)
            except Exception:
                pass

        for symbol, position in self.active_positions.items():
            try:
                side = 'SELL' if position.side == 'LONG' else 'BUY'
                self.adapter.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=position.size,
                    reduce_only=True
                )
                logger.info(f"Closed position: {side} {position.size} {symbol}")

                # Update realized P&L for the day when we can.
                if current_time is not None and current_price is not None:
                    entry_side = self._entry_side_by_symbol.get(symbol)
                    entry_price = self._entry_price_by_symbol.get(symbol)
                    entry_qty = self._entry_qty_by_symbol.get(symbol)
                    if entry_side and entry_price is not None and entry_qty is not None:
                        direction = 1.0 if str(entry_side).upper() == 'BUY' else -1.0
                        realized = (float(current_price) - float(entry_price)) * float(entry_qty) * direction
                        day_key = self._day_key(current_time)
                        self._realized_pnl_after_costs_per_day[day_key] = float(
                            self._realized_pnl_after_costs_per_day.get(day_key, 0.0)
                        ) + float(realized)

                # Cleanup local entry tracking.
                self._entry_side_by_symbol.pop(symbol, None)
                self._entry_price_by_symbol.pop(symbol, None)
                self._entry_qty_by_symbol.pop(symbol, None)
            except Exception as e:
                logger.error(f"Failed to close position {symbol}: {e}")
        
        # Cancel all stop orders
        for symbol, order_id in self.active_stop_orders.items():
            try:
                self.adapter.cancel_order(symbol, order_id)
            except Exception as e:
                logger.warning(f"Failed to cancel stop order {order_id}: {e}")
        
        self.active_positions.clear()
        self.active_stop_orders.clear()
        self.active_stop_prices.clear()

    def _apply_portfolio_open_risk_cap(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        quantity: float,
    ) -> Optional[float]:
        """Scale quantity down to satisfy portfolio max_open_risk_pct (risk-to-stop).

        Returns adjusted quantity, or 0/None if trade should be skipped.
        """
        portfolio_cfg = getattr(self.strategy.config, 'portfolio', None)
        if not portfolio_cfg or not getattr(portfolio_cfg, 'enabled', False):
            return quantity

        max_open_risk_pct = getattr(portfolio_cfg, 'max_open_risk_pct', None)
        if max_open_risk_pct is None or max_open_risk_pct <= 0:
            return quantity

        risk_cfg = getattr(self.strategy.config, 'risk', None)
        sizing_mode = getattr(risk_cfg, 'sizing_mode', 'account_size') if risk_cfg else 'account_size'

        if sizing_mode == 'account_size':
            base = float(getattr(risk_cfg, 'account_size', self.initial_capital) if risk_cfg else self.initial_capital)
        else:
            balance = self.adapter.get_account_balance()
            base = float(balance.total_balance)

        max_open_risk_amount = base * (float(max_open_risk_pct) / 100.0)

        current_open_risk = self._calculate_open_risk_amount_live()
        if current_open_risk is None:
            # Conservative: if we can't estimate, don't add exposure.
            logger.warning("Cannot estimate current open risk (missing stop tracking); blocking new entry")
            return None

        remaining = max_open_risk_amount - float(current_open_risk)
        if remaining <= 0:
            return 0.0

        proposed_risk = self._estimate_risk_to_stop_amount_for_symbol(symbol, entry_price, stop_price, quantity)
        if proposed_risk <= 0:
            return 0.0

        if proposed_risk <= remaining:
            return quantity

        scale = remaining / proposed_risk
        new_qty = quantity * scale

        sym_spec = self._get_market_spec_for_symbol(symbol)

        # Futures (CME/CBOT-style) cannot be scaled below 1 contract when the cap is active.
        # Match BacktestEngine: if scaled quantity < 1, skip; otherwise floor to whole contracts.
        if sym_spec.asset_class == 'futures' and sym_spec.contract_size is not None:
            if float(new_qty) < 1.0:
                return 0.0
            try:
                new_qty = float(np.floor(float(new_qty)))
            except Exception:
                new_qty = float(int(float(new_qty)))
            new_qty = max(1.0, float(new_qty))
            return float(new_qty)

        # Non-futures: Respect exchange precision/min size
        new_qty = round(float(new_qty), int(getattr(sym_spec, 'quantity_precision', 2)))
        if new_qty < float(getattr(sym_spec, 'min_trade_size', 0.0)):
            return 0.0
        return float(new_qty)

    def _calculate_open_risk_amount_live(self) -> Optional[float]:
        """Compute open risk across all active positions using tracked stop prices.

        Returns None if any open position lacks a known stop price.
        """
        total = 0.0
        for sym, pos in self.active_positions.items():
            sp = self.active_stop_prices.get(sym)
            if sp is None:
                return None
            entry_price = float(getattr(pos, 'entry_price', 0.0) or 0.0)
            if entry_price <= 0:
                entry_price = float(self._entry_price_by_symbol.get(sym, 0.0) or 0.0)
            total += self._estimate_risk_to_stop_amount_for_symbol(sym, float(entry_price), float(sp), float(pos.size))
        return float(total)

    def _estimate_risk_to_stop_amount_for_symbol(self, symbol: str, entry_price: float, stop_price: float, quantity: float) -> float:
        spec = self._get_market_spec_for_symbol(symbol)
        return self._estimate_risk_to_stop_amount_for_spec(spec, entry_price, stop_price, quantity)

    def _estimate_risk_to_stop_amount_for_spec(self, spec: MarketSpec, entry_price: float, stop_price: float, quantity: float) -> float:
        price_risk = abs(float(entry_price) - float(stop_price))
        if price_risk <= 0 or quantity <= 0:
            return 0.0

        if spec.asset_class == 'forex' and spec.contract_size is not None:
            pip_size = spec.pip_value or 0.0001
            stop_loss_pips = price_risk / pip_size
            pip_value_per_unit = spec.calculate_pip_value_per_unit(entry_price)
            if stop_loss_pips <= 0 or pip_value_per_unit <= 0:
                return 0.0
            return abs(quantity) * stop_loss_pips * pip_value_per_unit

        if spec.asset_class == 'futures' and spec.contract_size is not None:
            return abs(quantity) * price_risk * float(spec.contract_size)

        return abs(quantity) * price_risk

    def _estimate_risk_to_stop_amount(self, entry_price: float, stop_price: float, quantity: float) -> float:
        # Backward-compatible wrapper for single-symbol engines.
        return self._estimate_risk_to_stop_amount_for_spec(self.market_spec, entry_price, stop_price, quantity)

    def _get_stop_price_with_fallback_live(
        self,
        stop_price: Any,
        reference_price: float,
        dir_int: int,
        recent_bars: Optional[pd.DataFrame],
    ) -> Optional[float]:
        """Validate stop_price or compute ATR fallback when enabled."""
        if self._is_valid_stop(stop_price, reference_price, dir_int):
            return float(stop_price)

        portfolio_cfg = getattr(self.strategy.config, 'portfolio', None)
        if not portfolio_cfg or not getattr(portfolio_cfg, 'enabled', False):
            return None
        if not getattr(portfolio_cfg, 'stop_fallback_enabled', False):
            return None

        atr_period = int(getattr(portfolio_cfg, 'stop_fallback_atr_period', 14))
        atr_mult = float(getattr(portfolio_cfg, 'stop_fallback_atr_multiplier', 3.0))

        atr = self._calculate_atr_value(recent_bars, atr_period)
        if atr is None or not np.isfinite(atr) or atr <= 0:
            return None

        dist = atr_mult * atr
        fallback_stop = float(reference_price) - (float(dir_int) * dist)
        if self._is_valid_stop(fallback_stop, reference_price, dir_int):
            return float(fallback_stop)
        return None

    @staticmethod
    def _is_valid_stop(stop_price: Any, reference_price: float, dir_int: int) -> bool:
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
    def _calculate_atr_value(bars: Optional[pd.DataFrame], period: int) -> Optional[float]:
        """Compute ATR(period) from cached bars, using only past bars."""
        if bars is None or len(bars) < period + 1:
            return None
        if period <= 0:
            return None

        required_cols = {'high', 'low', 'close'}
        if not required_cols.issubset(bars.columns):
            return None

        window = bars.iloc[-(period + 1):]
        high = window['high'].astype(float)
        low = window['low'].astype(float)
        close = window['close'].astype(float)
        prev_close = close.shift(1)

        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr = tr.iloc[1:]
        if len(tr) < period:
            return None
        atr = tr.iloc[-period:].mean()
        if pd.isna(atr):
            return None
        return float(atr)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        balance = self.adapter.get_account_balance()
        
        return {
            'balance': balance.total_balance,
            'available': balance.available_balance,
            'margin_used': balance.margin_used,
            'unrealized_pnl': balance.unrealized_pnl,
            'positions': len(self.active_positions),
            'trades_executed': self.trades_executed,
            'trades_skipped': self.trades_skipped,
            'daily_pnl': balance.total_balance - self.daily_start_balance,
            'paper_mode': self.adapter.paper_mode
        }

