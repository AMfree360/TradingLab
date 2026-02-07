"""Volatility-based strategy template.

This template provides a base class for volatility-based strategies with
pre-configured filters optimized for volatile market conditions.
"""

from typing import Dict, Optional
import pandas as pd
from abc import abstractmethod
from strategies.base import StrategyBase
from config.schema import StrategyConfig


class VolatilityStrategy(StrategyBase):
    """Base class for volatility-based strategies.
    
    This template provides:
    - Pre-configured ATR-based filters
    - Volatility regime detection
    - Optimized for volatile market conditions
    
    Subclasses should implement:
    - get_indicators(): Calculate technical indicators
    - _generate_volatility_signal(): Generate volatility-based signals
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize volatility strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Volatility strategies typically need ATR filters
        if not config.regime_filters.atr_threshold.enabled:
            # Auto-enable ATR threshold for volatility strategies
            config.regime_filters.atr_threshold.enabled = True
    
    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate volatility-based signals with filter integration.
        
        This method:
        1. Checks time blackouts before generating signals
        2. Generates base volatility signals
        3. Applies filter chain to each signal
        
        Args:
            df_by_tf: Dictionary mapping timeframe strings to DataFrames
            
        Returns:
            DataFrame with filtered signals
        """
        signals = []
        signal_tf = self.config.timeframes.signal_tf
        symbol = self.config.market.symbol
        
        if signal_tf not in df_by_tf:
            return self._create_signal_dataframe()
        
        df_signal = df_by_tf[signal_tf]
        
        # Iterate through signal timeframe
        for idx, row in df_signal.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)
            
            # Check time blackout BEFORE generating signal
            if self._should_block_signal_generation(timestamp, symbol):
                continue  # Skip signal generation during blackout
            
            # Generate base volatility signal
            signal = self._generate_volatility_signal(row, df_by_tf, timestamp)
            if signal is None:
                continue
            
            # Apply filter chain
            if not self.apply_filters(
                signal=signal,
                timestamp=timestamp,
                symbol=symbol,
                df_by_tf=df_by_tf
            ):
                continue  # Signal filtered out
            
            # Add to signals list
            signals.append({
                'timestamp': timestamp,
                'direction': signal.get('direction', 'long'),
                'entry_price': signal.get('entry_price', row['close']),
                'stop_price': signal.get('stop_price'),
                'weight': signal.get('weight', 1.0),
                'metadata': signal.get('metadata', {})
            })
        
        if len(signals) == 0:
            return self._create_signal_dataframe()
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        signals_df.set_index('timestamp', inplace=True)
        return signals_df
    
    def _generate_volatility_signal(
        self,
        row: pd.Series,
        df_by_tf: Dict[str, pd.DataFrame],
        timestamp: pd.Timestamp
    ) -> Dict:
        """
        Generate a volatility-based signal.
        
        This is a template method that subclasses should override.
        
        Args:
            row: Current signal timeframe bar
            df_by_tf: All timeframe data
            timestamp: Current timestamp
            
        Returns:
            Signal dict with direction, entry_price, stop_price, etc., or None
        """
        # Template method - subclasses should implement
        return None
    
    @abstractmethod
    def get_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
        """
        Compute technical indicators for a DataFrame.
        
        Subclasses must implement this method.
        
        Args:
            df: DataFrame with OHLCV data and datetime index
            tf: Optional timeframe label (e.g., '1h', '15m')
            
        Returns:
            DataFrame with original data plus indicator columns
        """
        pass
