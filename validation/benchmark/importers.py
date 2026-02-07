"""Trade importers for external platforms.

Supports importing trades from:
- MT5 (MetaTrader 5)
- NinjaTrader
- TradingView

All importers convert external trade formats to Trading Lab's Trade dataclass.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from engine.backtest_engine import Trade


class TradeImporter(ABC):
    """Base class for trade importers."""
    
    @abstractmethod
    def import_trades(self, file_path: str) -> List[Trade]:
        """Import trades from a file.
        
        Args:
            file_path: Path to the trade export file
            
        Returns:
            List of Trade objects
        """
        pass
    
    @staticmethod
    @staticmethod
    def _parse_float(value, default=0.0):
        """Parse a value to float, handling currency formatting."""
        if pd.isna(value) or value == '' or value is None:
            return default
        try:
            # Convert to string and remove currency symbols, commas, spaces
            value_str = str(value).strip()
            value_str = value_str.replace('$', '').replace(',', '').replace(' ', '')
            # Remove parentheses (often used for negative values in accounting)
            if value_str.startswith('(') and value_str.endswith(')'):
                value_str = '-' + value_str[1:-1]
            return float(value_str)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _parse_timestamp(ts: str) -> pd.Timestamp:
        """Parse timestamp string to pd.Timestamp."""
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y.%m.%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%d.%m.%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%SZ',
        ]
        
        for fmt in formats:
            try:
                return pd.Timestamp(datetime.strptime(ts, fmt))
            except (ValueError, TypeError):
                continue
        
        # Fallback to pandas parser
        return pd.Timestamp(ts)


class MT5Importer(TradeImporter):
    """Importer for MT5 Strategy Tester trade exports.
    
    Expected CSV format (from MT5 Strategy Tester):
    - Time, Type, Entry, Volume, Price, S/L, T/P, Time, Price, Commission, Swap, Profit, Comment
    - Or similar columns with trade details
    """
    
    def import_trades(self, file_path: str) -> List[Trade]:
        """Import trades from MT5 export file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"MT5 trade file not found: {file_path}")
        
        # Try to read CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read MT5 trade file: {e}")
        
        trades = []
        
        # MT5 typically exports one row per trade with entry and exit info
        # Common column names (case-insensitive matching)
        df.columns = df.columns.str.strip().str.lower()
        
        # Map common MT5 column names
        entry_time_col = None
        exit_time_col = None
        entry_price_col = None
        exit_price_col = None
        volume_col = None
        type_col = None
        profit_col = None
        commission_col = None
        swap_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'entry' in col_lower and 'time' in col_lower:
                entry_time_col = col
            elif 'exit' in col_lower and 'time' in col_lower:
                exit_time_col = col
            elif 'time' in col_lower and entry_time_col is None:
                entry_time_col = col  # First time column
            elif 'entry' in col_lower and 'price' in col_lower:
                entry_price_col = col
            elif 'exit' in col_lower and 'price' in col_lower:
                exit_price_col = col
            elif 'price' in col_lower and entry_price_col is None:
                entry_price_col = col  # First price column
            elif 'volume' in col_lower or 'lots' in col_lower or 'size' in col_lower:
                volume_col = col
            elif 'type' in col_lower or 'direction' in col_lower:
                type_col = col
            elif 'profit' in col_lower:
                profit_col = col
            elif 'commission' in col_lower:
                commission_col = col
            elif 'swap' in col_lower:
                swap_col = col
        
        if entry_time_col is None or entry_price_col is None:
            raise ValueError(
                f"MT5 file missing required columns. Found: {list(df.columns)}\n"
                "Expected: entry time, entry price, volume/type"
            )
        
        for idx, row in df.iterrows():
            try:
                # Parse entry time
                entry_time = self._parse_timestamp(str(row[entry_time_col]))
                
                # Parse exit time (use entry time if not available)
                if exit_time_col and exit_time_col in df.columns:
                    exit_time = self._parse_timestamp(str(row[exit_time_col]))
                else:
                    exit_time = entry_time  # Assume same bar exit
                
                # Parse direction
                if type_col and type_col in df.columns:
                    type_val = str(row[type_col]).lower()
                    if 'buy' in type_val or 'long' in type_val:
                        direction = 'long'
                    elif 'sell' in type_val or 'short' in type_val:
                        direction = 'short'
                    else:
                        direction = 'long'  # Default
                else:
                    direction = 'long'  # Default
                
                # Parse prices
                entry_price = float(row[entry_price_col])
                if exit_price_col and exit_price_col in df.columns:
                    exit_price = float(row[exit_price_col])
                else:
                    # If no exit price, try to infer from profit
                    if profit_col and profit_col in df.columns:
                        profit = float(row[profit_col])
                        if direction == 'long':
                            exit_price = entry_price + (profit / (row[volume_col] if volume_col else 1.0))
                        else:
                            exit_price = entry_price - (profit / (row[volume_col] if volume_col else 1.0))
                    else:
                        exit_price = entry_price
                
                # Parse volume (MT5 uses lots, convert to units)
                if volume_col and volume_col in df.columns:
                    volume_lots = float(row[volume_col])
                    # Assume standard lot size (100,000 for forex, adjust as needed)
                    quantity = volume_lots * 100000.0
                else:
                    quantity = 1.0
                
                # Parse P&L
                pnl_raw = 0.0
                if profit_col and profit_col in df.columns:
                    pnl_raw = float(row[profit_col])
                
                # Parse commission
                commission = 0.0
                if commission_col and commission_col in df.columns:
                    commission = float(row[commission_col])
                
                # Parse swap
                swap = 0.0
                if swap_col and swap_col in df.columns:
                    swap = float(row[swap_col])
                
                # Calculate P&L after costs
                pnl_after_costs = pnl_raw - commission - swap
                
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl_raw=pnl_raw,
                    pnl_after_costs=pnl_after_costs,
                    commission=commission,
                    slippage=0.0,  # MT5 doesn't always export this separately
                    stop_price=None,
                    exit_reason=None,
                )
                
                trades.append(trade)
                
            except Exception as e:
                print(f"Warning: Skipped row {idx} in MT5 file: {e}")
                continue
        
        return trades


class NinjaTraderImporter(TradeImporter):
    """Importer for NinjaTrader Strategy Analyzer exports.
    
    Expected CSV format:
    - Entry Time, Exit Time, Entry Price, Exit Price, Quantity, Direction, P&L, Commission, etc.
    """
    
    def import_trades(self, file_path: str) -> List[Trade]:
        """Import trades from NinjaTrader export file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"NinjaTrader trade file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read NinjaTrader trade file: {e}")
        
        trades = []
        df.columns = df.columns.str.strip().str.lower()
        
        # Map NinjaTrader column names
        entry_time_col = None
        exit_time_col = None
        entry_price_col = None
        exit_price_col = None
        quantity_col = None
        direction_col = None
        pnl_col = None
        commission_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'entry' in col_lower and 'time' in col_lower:
                entry_time_col = col
            elif 'exit' in col_lower and 'time' in col_lower:
                exit_time_col = col
            elif 'entry' in col_lower and 'price' in col_lower:
                entry_price_col = col
            elif 'exit' in col_lower and 'price' in col_lower:
                exit_price_col = col
            elif 'quantity' in col_lower or 'size' in col_lower or 'contracts' in col_lower:
                quantity_col = col
            elif 'direction' in col_lower or 'side' in col_lower:
                direction_col = col
            elif 'pnl' in col_lower or 'profit' in col_lower or 'net' in col_lower:
                pnl_col = col
            elif 'commission' in col_lower:
                commission_col = col
        
        # Fallback: use generic 'time' and 'price' if entry/exit specific columns not found
        if entry_time_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'time' or col_lower == 'datetime' or col_lower == 'date':
                    entry_time_col = col
                    break
        
        if entry_price_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if col_lower == 'price':
                    entry_price_col = col
                    break
        
        if entry_time_col is None or entry_price_col is None:
            raise ValueError(
                f"NinjaTrader file missing required columns. Found: {list(df.columns)}\n"
                f"Need at least 'time' (or 'entry_time') and 'price' (or 'entry_price') columns."
            )
        
        # Detect action column for direction inference
        action_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'action' or col_lower == 'e/x':
                action_col = col
                break
        
        # Treat each row as a complete trade (or entry/exit will be paired later if needed)
        for idx, row in df.iterrows():
            try:
                entry_time = self._parse_timestamp(str(row[entry_time_col]))
                exit_time = self._parse_timestamp(
                    str(row[exit_time_col]) if exit_time_col and exit_time_col in df.columns 
                    else str(row[entry_time_col])
                )
                
                entry_price = self._parse_float(row[entry_price_col])
                exit_price = self._parse_float(
                    row[exit_price_col] if exit_price_col and exit_price_col in df.columns else row[entry_price_col]
                )
                
                # Infer direction from action column if available
                if action_col and action_col in df.columns:
                    action_val = str(row[action_col]).lower()
                    if 'buy' in action_val or 'long' in action_val or 'enterlong' in action_val:
                        direction = 'long'
                    elif 'sell' in action_val or 'short' in action_val or 'entershort' in action_val:
                        direction = 'short'
                    else:
                        direction = 'long'  # Default
                elif direction_col and direction_col in df.columns:
                    dir_val = str(row[direction_col]).lower()
                    direction = 'long' if 'long' in dir_val or 'buy' in dir_val else 'short'
                else:
                    direction = 'long'  # Default
                
                quantity = self._parse_float(row[quantity_col], default=1.0) if quantity_col and quantity_col in df.columns else 1.0
                
                pnl_raw = self._parse_float(row[pnl_col]) if pnl_col and pnl_col in df.columns else 0.0
                commission = self._parse_float(row[commission_col]) if commission_col and commission_col in df.columns else 0.0
                
                pnl_after_costs = pnl_raw - commission
                
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl_raw=pnl_raw,
                    pnl_after_costs=pnl_after_costs,
                    commission=commission,
                    slippage=0.0,
                    stop_price=None,
                    exit_reason=None,
                )
                
                trades.append(trade)
                
            except Exception as e:
                print(f"Warning: Skipped row {idx} in NinjaTrader file: {e}")
                continue
        
        return trades


class TradingViewImporter(TradeImporter):
    """Importer for TradingView Strategy Tester exports.
    
    Expected CSV format:
    - Entry Time, Exit Time, Entry Price, Exit Price, Quantity, Direction, P&L, etc.
    """
    
    def import_trades(self, file_path: str) -> List[Trade]:
        """Import trades from TradingView export file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"TradingView trade file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read TradingView trade file: {e}")
        
        trades = []
        df.columns = df.columns.str.strip().str.lower()
        
        # TradingView column mapping
        entry_time_col = None
        exit_time_col = None
        entry_price_col = None
        exit_price_col = None
        quantity_col = None
        direction_col = None
        pnl_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'entry' in col_lower and 'time' in col_lower:
                entry_time_col = col
            elif 'exit' in col_lower and 'time' in col_lower:
                exit_time_col = col
            elif 'entry' in col_lower and 'price' in col_lower:
                entry_price_col = col
            elif 'exit' in col_lower and 'price' in col_lower:
                exit_price_col = col
            elif 'quantity' in col_lower or 'size' in col_lower or 'qty' in col_lower:
                quantity_col = col
            elif 'direction' in col_lower or 'side' in col_lower or 'type' in col_lower:
                direction_col = col
            elif 'pnl' in col_lower or 'profit' in col_lower or 'net' in col_lower:
                pnl_col = col
        
        if entry_time_col is None or entry_price_col is None:
            raise ValueError(
                f"TradingView file missing required columns. Found: {list(df.columns)}"
            )
        
        for idx, row in df.iterrows():
            try:
                entry_time = self._parse_timestamp(str(row[entry_time_col]))
                exit_time = self._parse_timestamp(
                    str(row[exit_time_col]) if exit_time_col and exit_time_col in df.columns 
                    else str(row[entry_time_col])
                )
                
                entry_price = float(row[entry_price_col])
                exit_price = float(row[exit_price_col]) if exit_price_col and exit_price_col in df.columns else entry_price
                
                if direction_col and direction_col in df.columns:
                    dir_val = str(row[direction_col]).lower()
                    direction = 'long' if 'long' in dir_val or 'buy' in dir_val else 'short'
                else:
                    direction = 'long'
                
                quantity = float(row[quantity_col]) if quantity_col and quantity_col in df.columns else 1.0
                
                pnl_raw = float(row[pnl_col]) if pnl_col and pnl_col in df.columns else 0.0
                commission = 0.0  # TradingView often includes commission in P&L
                
                pnl_after_costs = pnl_raw - commission
                
                trade = Trade(
                    entry_time=entry_time,
                    exit_time=exit_time,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl_raw=pnl_raw,
                    pnl_after_costs=pnl_after_costs,
                    commission=commission,
                    slippage=0.0,
                    stop_price=None,
                    exit_reason=None,
                )
                
                trades.append(trade)
                
            except Exception as e:
                print(f"Warning: Skipped row {idx} in TradingView file: {e}")
                continue
        
        return trades


def import_trades_from_file(
    file_path: str,
    platform: Optional[str] = None
) -> List[Trade]:
    """Auto-detect platform and import trades from file.
    
    Args:
        file_path: Path to trade export file
        platform: Platform name ('mt5', 'ninjatrader', 'tradingview'). 
                  If None, will try to auto-detect from filename/content.
    
    Returns:
        List of Trade objects
    """
    path = Path(file_path)
    
    if platform is None:
        # Auto-detect from filename
        filename_lower = path.name.lower()
        if 'mt5' in filename_lower or 'metatrader' in filename_lower:
            platform = 'mt5'
        elif 'ninja' in filename_lower or 'ninjatrader' in filename_lower:
            platform = 'ninjatrader'
        elif 'tradingview' in filename_lower or 'tv' in filename_lower:
            platform = 'tradingview'
        else:
            # Default to MT5 (most common)
            platform = 'mt5'
    
    platform = platform.lower()
    
    if platform == 'mt5':
        importer = MT5Importer()
    elif platform == 'ninjatrader':
        importer = NinjaTraderImporter()
    elif platform == 'tradingview':
        importer = TradingViewImporter()
    else:
        raise ValueError(f"Unknown platform: {platform}. Use 'mt5', 'ninjatrader', or 'tradingview'")
    
    return importer.import_trades(file_path)
