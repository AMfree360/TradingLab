"""Binance Futures API adapter for live trading.

Supports both paper trading (simulated) and live trading modes.
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    Client = None
    BinanceAPIException = Exception

logger = logging.getLogger(__name__)

# Load .env file if available
if DOTENV_AVAILABLE:
    # Try to find .env file in project root (2 levels up from this file)
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded .env file from {env_path}")
    else:
        # Try current directory
        load_dotenv()
        logger.debug("Attempted to load .env from current directory")


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET', 'LIMIT', 'STOP_MARKET', etc.
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = 'NEW'  # NEW, FILLED, CANCELED, REJECTED
    filled_quantity: float = 0.0
    avg_price: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class Position:
    """Position representation."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    size: float
    entry_price: float
    unrealized_pnl: float = 0.0
    leverage: float = 1.0
    margin: float = 0.0


@dataclass
class AccountBalance:
    """Account balance representation."""
    total_balance: float
    available_balance: float
    margin_used: float = 0.0
    unrealized_pnl: float = 0.0


class BinanceFuturesAdapter:
    """Adapter for Binance Futures API.
    
    Supports both paper trading (simulated) and live trading modes.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper_mode: bool = True,
        testnet: bool = True
    ):
        """
        Initialize Binance Futures adapter.
        
        Args:
            api_key: Binance API key (from environment if None)
            api_secret: Binance API secret (from environment if None)
            paper_mode: If True, simulates orders without placing real orders
            testnet: If True, uses Binance testnet (only for live mode)
        """
        if not BINANCE_AVAILABLE:
            raise ImportError(
                "python-binance package not installed. "
                "Install with: pip install python-binance"
            )
        
        self.paper_mode = paper_mode
        self.testnet = testnet
        
        # Get credentials from environment if not provided
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        # Debug: Check if credentials were found (without exposing secrets)
        if not self.paper_mode:
            if not self.api_key:
                logger.error("BINANCE_API_KEY not found in environment variables")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Looking for .env file in: {Path(__file__).parent.parent.parent / '.env'}")
                raise ValueError(
                    "API credentials required for live trading. "
                    "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables or in .env file."
                )
            if not self.api_secret:
                logger.error("BINANCE_API_SECRET not found in environment variables")
                raise ValueError(
                    "API credentials required for live trading. "
                    "Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables or in .env file."
                )
            logger.info(f"✓ API credentials loaded (key: {self.api_key[:8]}...{self.api_key[-4:] if len(self.api_key) > 12 else '***'})")
            
            # Initialize Binance client
            if testnet:
                self.client = Client(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=True
                )
                logger.info("Using Binance Testnet for live trading")
            else:
                self.client = Client(
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                logger.warning("Using Binance LIVE API - real money at risk!")
        else:
            self.client = None
            logger.info("Paper trading mode enabled - no real orders will be placed")
        
        # Paper trading state
        self._paper_balance = 10000.0  # Default paper balance
        self._paper_positions: Dict[str, Position] = {}
        self._paper_orders: Dict[str, Order] = {}
        self._paper_order_counter = 0
        
    def get_account_balance(self) -> AccountBalance:
        """Get account balance."""
        if self.paper_mode:
            # Calculate paper balance
            total_pnl = sum(pos.unrealized_pnl for pos in self._paper_positions.values())
            margin_used = sum(pos.margin for pos in self._paper_positions.values())
            
            return AccountBalance(
                total_balance=self._paper_balance + total_pnl,
                available_balance=self._paper_balance - margin_used + total_pnl,
                margin_used=margin_used,
                unrealized_pnl=total_pnl
            )
        else:
            # Get real balance from Binance
            account = self.client.futures_account()
            balance = float(account['totalWalletBalance'])
            available = float(account['availableBalance'])
            margin_used = float(account['totalMarginBalance']) - available
            
            return AccountBalance(
                total_balance=balance,
                available_balance=available,
                margin_used=margin_used,
                unrealized_pnl=0.0  # Binance provides this separately
            )
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions."""
        if self.paper_mode:
            positions = list(self._paper_positions.values())
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]
            return positions
        else:
            # Get real positions from Binance
            if symbol:
                positions_data = self.client.futures_position_information(symbol=symbol)
            else:
                positions_data = self.client.futures_position_information()
            
            positions = []
            for pos_data in positions_data:
                size = float(pos_data['positionAmt'])
                if abs(size) > 0.0001:  # Only non-zero positions
                    entry_price = float(pos_data['entryPrice'])
                    unrealized_pnl = float(pos_data['unRealizedProfit'])
                    leverage = float(pos_data.get('leverage', 1))
                    
                    positions.append(Position(
                        symbol=pos_data['symbol'],
                        side='LONG' if size > 0 else 'SHORT',
                        size=abs(size),
                        entry_price=entry_price,
                        unrealized_pnl=unrealized_pnl,
                        leverage=leverage,
                        margin=0.0  # Calculate separately if needed
                    ))
            
            return positions
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float,
        reduce_only: bool = False
    ) -> Order:
        """Place a market order."""
        if self.paper_mode:
            return self._place_paper_market_order(symbol, side, quantity)
        else:
            return self._place_real_market_order(symbol, side, quantity, reduce_only)
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = 'GTC',  # GTC, IOC, FOK
        reduce_only: bool = False
    ) -> Order:
        """Place a limit order."""
        if self.paper_mode:
            return self._place_paper_limit_order(symbol, side, quantity, price)
        else:
            return self._place_real_limit_order(
                symbol, side, quantity, price, time_in_force, reduce_only
            )
    
    def place_stop_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True
    ) -> Order:
        """Place a stop market order (stop loss)."""
        if self.paper_mode:
            return self._place_paper_stop_order(symbol, side, quantity, stop_price)
        else:
            return self._place_real_stop_order(symbol, side, quantity, stop_price, reduce_only)
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        if self.paper_mode:
            if order_id in self._paper_orders:
                order = self._paper_orders[order_id]
                order.status = 'CANCELED'
                return True
            return False
        else:
            try:
                self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
                return True
            except BinanceAPIException as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                return False
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        if self.paper_mode:
            # In paper mode, we need to fetch real price for simulation
            # Use public API (no auth needed)
            try:
                from binance.client import Client as PublicClient
                public_client = PublicClient()
                ticker = public_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            except Exception as e:
                logger.error(f"Failed to get price for {symbol}: {e}")
                raise
        else:
            # Use the client (works for both testnet and live)
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
    
    def _place_paper_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Order:
        """Place a simulated market order."""
        self._paper_order_counter += 1
        order_id = f"PAPER_{self._paper_order_counter}"
        
        current_price = self.get_current_price(symbol)
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type='MARKET',
            quantity=quantity,
            price=current_price,
            status='FILLED',
            filled_quantity=quantity,
            avg_price=current_price,
            timestamp=datetime.now()
        )
        
        # Update position
        self._update_paper_position(symbol, side, quantity, current_price)
        
        self._paper_orders[order_id] = order
        logger.info(f"PAPER: {side} {quantity} {symbol} @ {current_price}")
        
        return order
    
    def _place_paper_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Order:
        """Place a simulated limit order."""
        self._paper_order_counter += 1
        order_id = f"PAPER_{self._paper_order_counter}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type='LIMIT',
            quantity=quantity,
            price=price,
            status='NEW',
            timestamp=datetime.now()
        )
        
        self._paper_orders[order_id] = order
        logger.info(f"PAPER: LIMIT {side} {quantity} {symbol} @ {price}")
        
        # In paper mode, we could simulate limit order fills based on price movement
        # For now, we'll just create the order
        
        return order
    
    def _place_paper_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float
    ) -> Order:
        """Place a simulated stop order."""
        self._paper_order_counter += 1
        order_id = f"PAPER_{self._paper_order_counter}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type='STOP_MARKET',
            quantity=quantity,
            stop_price=stop_price,
            status='NEW',
            timestamp=datetime.now()
        )
        
        self._paper_orders[order_id] = order
        logger.info(f"PAPER: STOP {side} {quantity} {symbol} @ {stop_price}")
        
        return order
    
    def _update_paper_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ):
        """Update paper trading position."""
        if symbol not in self._paper_positions:
            # New position
            self._paper_positions[symbol] = Position(
                symbol=symbol,
                side='LONG' if side == 'BUY' else 'SHORT',
                size=quantity,
                entry_price=price,
                leverage=1.0,
                margin=0.0
            )
        else:
            # Update existing position
            pos = self._paper_positions[symbol]
            if pos.side == ('LONG' if side == 'BUY' else 'SHORT'):
                # Same direction - increase size
                total_value = pos.size * pos.entry_price + quantity * price
                pos.size += quantity
                pos.entry_price = total_value / pos.size
            else:
                # Opposite direction - reduce or reverse
                if quantity >= pos.size:
                    # Reverse position
                    pos.size = quantity - pos.size
                    pos.entry_price = price
                    pos.side = 'LONG' if side == 'BUY' else 'SHORT'
                else:
                    # Reduce position
                    pos.size -= quantity
    
    def _place_real_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool
    ) -> Order:
        """Place a real market order on Binance."""
        try:
            order_response = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                reduceOnly=reduce_only
            )
            
            order = Order(
                order_id=str(order_response['orderId']),
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=float(order_response['origQty']),
                status=order_response['status'],
                filled_quantity=float(order_response.get('executedQty', 0)),
                avg_price=float(order_response.get('avgPrice', 0)) if order_response.get('avgPrice') else None,
                timestamp=datetime.fromtimestamp(order_response['updateTime'] / 1000)
            )
            
            logger.info(f"LIVE: {side} {quantity} {symbol} - Order ID: {order.order_id}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Failed to place market order: {e}")
            raise
    
    def _place_real_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str,
        reduce_only: bool
    ) -> Order:
        """Place a real limit order on Binance."""
        try:
            order_response = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce=time_in_force,
                quantity=quantity,
                price=price,
                reduceOnly=reduce_only
            )
            
            order = Order(
                order_id=str(order_response['orderId']),
                symbol=symbol,
                side=side,
                order_type='LIMIT',
                quantity=float(order_response['origQty']),
                price=price,
                status=order_response['status'],
                timestamp=datetime.fromtimestamp(order_response['updateTime'] / 1000)
            )
            
            logger.info(f"LIVE: LIMIT {side} {quantity} {symbol} @ {price} - Order ID: {order.order_id}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Failed to place limit order: {e}")
            raise
    
    def _place_real_stop_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool
    ) -> Order:
        """Place a real stop market order on Binance."""
        try:
            # Binance uses STOP_MARKET for stop loss orders
            order_response = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                reduceOnly=reduce_only
            )
            
            order = Order(
                order_id=str(order_response['orderId']),
                symbol=symbol,
                side=side,
                order_type='STOP_MARKET',
                quantity=float(order_response['origQty']),
                stop_price=stop_price,
                status=order_response['status'],
                timestamp=datetime.fromtimestamp(order_response['updateTime'] / 1000)
            )
            
            logger.info(f"LIVE: STOP {side} {quantity} {symbol} @ {stop_price} - Order ID: {order.order_id}")
            return order
            
        except BinanceAPIException as e:
            logger.error(f"Failed to place stop order: {e}")
            raise

