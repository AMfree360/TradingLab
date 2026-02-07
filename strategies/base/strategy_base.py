"""Base strategy interface that all strategies must implement."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, TYPE_CHECKING
import pandas as pd
from config.schema import StrategyConfig
from strategies.filters import FilterManager
from strategies.filters.base import FilterContext
import yaml
from pathlib import Path

if TYPE_CHECKING:
    from engine.market import MarketSpec


class StrategyBase(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy with configuration.
        
        Args:
            config: Validated strategy configuration
        """
        self.config = config
        self.name = config.strategy_name
        
        # Do NOT initialize filters by default. Strategy implementations
        # or higher-level code can initialize filters explicitly if desired.
        self.filter_manager: Optional[FilterManager] = None

    def _filters_enabled_in_config(self) -> bool:
        """Return True if filters should be active for this strategy."""
        try:
            cal_cfg = getattr(self.config, 'calendar_filters', None)
            if cal_cfg is None:
                return False
            return bool(getattr(cal_cfg, 'master_filters_enabled', False))
        except Exception:
            return False
    
    @abstractmethod
    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals from multi-timeframe data.
        
        Args:
            df_by_tf: Dictionary mapping timeframe strings to DataFrames.
                     Each DataFrame should have datetime index and OHLCV columns:
                     ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with columns:
            - timestamp: datetime index
            - direction: 'long' or 'short'
            - entry_price: float (price to enter at)
            - stop_price: float (stop loss price)
            - weight: float (signal strength, 0.0 to 1.0, optional)
            - metadata: dict (additional signal info, optional)
        """
        pass
    
    @abstractmethod
    def get_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
        """
        Compute technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data and datetime index
            tf: Optional timeframe label (e.g., '1h', '15m')
        
        Returns:
            DataFrame with original data plus indicator columns
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate strategy configuration.
        
        Returns:
            True if config is valid, raises exception otherwise
        """
        # Basic validation - can be overridden by subclasses
        if not self.config.strategy_name:
            raise ValueError("Strategy name is required")
        return True
    
    def get_required_timeframes(self) -> list[str]:
        """
        Get list of required timeframes for this strategy.
        
        Returns:
            List of timeframe strings (e.g., ['1h', '15m'])
        """
        return [
            self.config.timeframes.signal_tf,
            self.config.timeframes.entry_tf,
        ]
    
    def prepare_data(self, df_by_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Prepare data by computing indicators for each timeframe.
        
        Args:
            df_by_tf: Dictionary mapping timeframe strings to DataFrames
        
        Returns:
            Dictionary with indicators computed for each timeframe
        """
        prepared = {}
        for tf, df in df_by_tf.items():
            prepared[tf] = self.get_indicators(df.copy(), tf=tf)
        return prepared
    
    def _validate_dataframe(self, df: pd.DataFrame, required_cols: list[str] = None) -> None:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            required_cols: List of required column names (default: OHLCV)
        
        Raises:
            ValueError if required columns are missing
        """
        if required_cols is None:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
    
    def _create_signal_dataframe(self) -> pd.DataFrame:
        """
        Create empty signal DataFrame with correct schema.
        
        Returns:
            Empty DataFrame with signal columns
        """
        return pd.DataFrame(columns=[
            'timestamp',
            'direction',
            'entry_price',
            'stop_price',
            'weight',
            'metadata'
        ]).set_index('timestamp')
    
    def _initialize_filters(self) -> None:
        """
        Initialize filter manager with master and strategy configs.
        
        This method loads the master filter configuration and merges it with
        strategy-specific filter configuration.
        """
        # Load master config if available
        master_config = None
        master_config_path = Path(__file__).parent.parent.parent / 'config' / 'master_filters.yml'
        if master_config_path.exists():
            try:
                with open(master_config_path, 'r') as f:
                    master_data = yaml.safe_load(f)
                    master_config = master_data.get('master_filters', {})
            except Exception:
                # If master config can't be loaded, continue without it
                master_config = None
        
        # Create filter manager
        self.filter_manager = FilterManager(
            master_config=master_config,
            strategy_config=self.config
        )
    
    def _should_block_signal_generation(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if signal generation should be blocked at this timestamp.
        
        This method checks time blackout filters BEFORE generating signals,
        allowing early exit from signal generation loops.
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            symbol: Symbol to check
            
        Returns:
            True if signal generation should be blocked
        """
        if not self._filters_enabled_in_config():
            return False

        if not self.filter_manager:
            self._initialize_filters()

        if not self.filter_manager:
            return False
        
        # Check time blackout filters
        for filter_obj in self.filter_manager.filters:
            if hasattr(filter_obj, 'should_block_signal_generation'):
                if filter_obj.should_block_signal_generation(timestamp, symbol):
                    return True
        
        return False
    
    def apply_filters(
        self,
        signal: pd.Series,
        timestamp: pd.Timestamp,
        symbol: str,
        df_by_tf: Dict[str, pd.DataFrame],
        market_spec: Optional["MarketSpec"] = None
    ) -> bool:
        """
        Apply filter chain to a signal.
        
        This is a convenience method that creates a FilterContext and applies
        all enabled filters to the signal.
        
        Args:
            signal: Signal Series with direction, entry_price, etc.
            timestamp: Signal timestamp
            symbol: Trading symbol
            df_by_tf: All timeframe data
            market_spec: Optional MarketSpec (auto-loaded if None)
            
        Returns:
            True if signal passed all filters, False otherwise
        """
        if not self._filters_enabled_in_config():
            return True

        if not self.filter_manager:
            self._initialize_filters()

        if not self.filter_manager:
            return True  # Conservative: if filters failed to init, don't block
        
        # Load MarketSpec if not provided
        if market_spec is None:
            # Import here to avoid circular import at module level
            from engine.market import MarketSpec
            try:
                market_spec = MarketSpec.load_from_profiles(symbol)
            except (ValueError, FileNotFoundError):
                # Fallback: create basic MarketSpec
                market_spec = MarketSpec(
                    symbol=symbol,
                    exchange='unknown',
                    asset_class='crypto'  # Default fallback
                )
        
        # Create filter context
        context = FilterContext(
            timestamp=timestamp,
            symbol=symbol,
            signal_direction=1 if signal.get('direction') == 'long' else -1,
            signal_data=signal,
            df_by_tf=df_by_tf,
            market_spec=market_spec
        )
        
        # Apply filters
        result = self.filter_manager.apply_filters(signal, context)
        return result.passed

    # Backward-compatibility stubs ---------------------------------------
    def _check_ema_alignment(self, entry_row, entry_ma_cfg, dir_int) -> bool:
        """
        Default EMA alignment check. Strategies may override this to
        implement entry-timeframe EMA gating logic. Returning True
        bypasses EMA alignment gating and allows entries by default.
        """
        return True

    def _check_macd_progression(self, df_entry, entry_df_idx, dir_int, min_bars) -> bool:
        """
        Default MACD progression check. Strategies may override this
        to require a minimum MACD progression before entry. Returns
        True by default to avoid rejecting signals when not implemented.
        """
        return True

    def _calculate_stop_loss(self, entry_row, dir_int: int) -> float:
        """
        Default stop-loss calculation.

        - Tries to read a percentage from `self.config.stop_loss.percent` or
          a strategy attribute `self.stoploss_pct` set by convenience.
        - Interprets values > 1 as percentage (e.g., 1.0 -> 1%), values <=1 as fraction (0.01 -> 1%).
        - Falls back to 1% if nothing provided.

        Returns stop price (float) or None if cannot compute.
        """
        try:
            entry_price = float(entry_row['close'])
        except Exception:
            return None

        # Try config value
        percent = None
        try:
            sl_cfg = getattr(self.config, 'stop_loss', None)
            if sl_cfg is not None and hasattr(sl_cfg, 'percent'):
                percent = getattr(sl_cfg, 'percent')
        except Exception:
            percent = None

        # Fallback to instance attr
        if percent is None:
            percent = getattr(self, 'stoploss_pct', None)

        # Default
        if percent is None:
            pct = 0.01
        else:
            try:
                pct = float(percent)
            except Exception:
                pct = 0.01

        # Interpret >1 as percent value (e.g., 1.0 -> 1%), convert to fraction
        if pct > 1:
            pct = pct / 100.0

        if dir_int == 1:
            return entry_price * (1.0 - pct)
        else:
            return entry_price * (1.0 + pct)


