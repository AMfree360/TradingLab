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
        
    def check_safety_limits(self) -> Tuple[bool, Optional[str]]:
        """Check if safety limits are exceeded.
        
        Returns:
            (should_stop, reason)
        """
        balance = self.adapter.get_account_balance()
        current_equity = balance.total_balance
        
        # Check daily loss
        daily_loss_pct = ((self.daily_start_balance - current_equity) / self.daily_start_balance) * 100
        if daily_loss_pct >= self.max_daily_loss_pct:
            return True, f"Daily loss limit exceeded: {daily_loss_pct:.2f}%"
        
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
        
        # Handle different actions
        if action == 'CLOSE':
            self._close_all_positions()
        elif action in ['BUY', 'SELL']:
            self._process_entry_signal(signal, current_price, current_time)
        elif action == 'UPDATE_STOP':
            self._update_stop_loss(signal, current_price)

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
        quantity = signal.get('quantity')
        stop_price = signal.get('stop_price')
        target_price = signal.get('target_price')
        
        if not quantity or quantity <= 0:
            logger.warning(f"Invalid quantity in signal: {quantity}")
            return
        
        # Check if we already have a position
        if self.symbol in self.active_positions:
            logger.info(f"Position already exists for {self.symbol}, skipping entry")
            self.trades_skipped += 1
            return

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
        
        # Check balance
        balance = self.adapter.get_account_balance()
        if balance.available_balance <= 0:
            logger.warning("Insufficient balance for trade")
            self.trades_skipped += 1
            return
        
        # Calculate margin required
        margin_required = self.market_spec.calculate_margin(current_price, quantity)
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
            
            # Place stop loss if provided
            if stop_price_final is not None:
                self._place_stop_loss(quantity, float(stop_price_final), side)
            
        except Exception as e:
            logger.error(f"Failed to place entry order: {e}")
            self.trades_skipped += 1
    
    def _place_stop_loss(self, quantity: float, stop_price: float, entry_side: str):
        """Place a stop loss order."""
        try:
            # Stop order side is opposite of entry
            stop_side = 'SELL' if entry_side == 'BUY' else 'BUY'
            
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
            stop_order = self.adapter.place_stop_market_order(
                symbol=self.symbol,
                side=stop_side,
                quantity=position.size,
                stop_price=new_stop_price,
                reduce_only=True
            )
            
            self.active_stop_orders[self.symbol] = stop_order.order_id
            self.active_stop_prices[self.symbol] = float(new_stop_price)
            logger.info(f"Stop loss updated: {stop_side} {position.size} {self.symbol} @ {new_stop_price}")
            
        except Exception as e:
            logger.error(f"Failed to update stop loss: {e}")
    
    def _close_all_positions(self):
        """Close all open positions."""
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

        proposed_risk = self._estimate_risk_to_stop_amount(entry_price, stop_price, quantity)
        if proposed_risk <= 0:
            return 0.0

        if proposed_risk <= remaining:
            return quantity

        scale = remaining / proposed_risk
        new_qty = quantity * scale

        # Respect exchange precision/min size
        new_qty = round(float(new_qty), int(getattr(self.market_spec, 'quantity_precision', 2)))
        if new_qty < float(getattr(self.market_spec, 'min_trade_size', 0.0)):
            return 0.0
        return new_qty

    def _calculate_open_risk_amount_live(self) -> Optional[float]:
        """Compute open risk across all active positions using tracked stop prices.

        Returns None if any open position lacks a known stop price.
        """
        total = 0.0
        for sym, pos in self.active_positions.items():
            sp = self.active_stop_prices.get(sym)
            if sp is None:
                return None
            total += self._estimate_risk_to_stop_amount(float(pos.entry_price), float(sp), float(pos.size))
        return float(total)

    def _estimate_risk_to_stop_amount(self, entry_price: float, stop_price: float, quantity: float) -> float:
        price_risk = abs(float(entry_price) - float(stop_price))
        if price_risk <= 0 or quantity <= 0:
            return 0.0

        # Binance-style perpetuals behave like spot: pnl = price_change * qty
        if self.market_spec.asset_class == 'forex' and self.market_spec.contract_size is not None:
            pip_size = self.market_spec.pip_value or 0.0001
            stop_loss_pips = price_risk / pip_size
            pip_value_per_unit = self.market_spec.calculate_pip_value_per_unit(entry_price)
            if stop_loss_pips <= 0 or pip_value_per_unit <= 0:
                return 0.0
            return abs(quantity) * stop_loss_pips * pip_value_per_unit

        if self.market_spec.asset_class == 'futures' and self.market_spec.contract_size is not None:
            return abs(quantity) * price_risk * float(self.market_spec.contract_size)

        return abs(quantity) * price_risk

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

