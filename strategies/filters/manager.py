"""Filter manager for applying filter chain to signals."""

from typing import List, Dict, Optional
from strategies.filters.base import FilterBase, FilterContext, FilterResult
from config.schema import StrategyConfig
import pandas as pd


class FilterManager:
    """Manages and applies filter chain to signals.
    
    The FilterManager builds a chain of filters from configuration and applies
    them sequentially to signals. Filters are applied in order, and the first
    filter that fails causes the signal to be rejected (short-circuiting).
    
    Example:
        manager = FilterManager(master_config, strategy_config)
        result = manager.apply_filters(signal, context)
        if result.passed:
            # Signal passed all filters
            pass
    """
    
    def __init__(self, master_config: Optional[Dict] = None, strategy_config: Optional[StrategyConfig] = None):
        """
        Initialize filter manager.
        
        Args:
            master_config: Master filter configuration (from config/master_filters.yml)
            strategy_config: Strategy-specific configuration
        """
        self.master_config = master_config or {}
        self.strategy_config = strategy_config
        self.filters: List[FilterBase] = []
        
        # Build filter chain if configs provided
        if master_config or strategy_config:
            self.filters = self._build_filter_chain()
    
    def _build_filter_chain(self) -> List[FilterBase]:
        """Build filter chain from merged configs.
        
        ARCHITECTURE: Filters are always built if configuration exists, regardless of master_filters_enabled.
        The master_filters_enabled flag controls whether filters are APPLIED in the strategy, not whether
        they are BUILT. This ensures filters are available for application when needed.
        """
        filters = []
        
        # Merge master and strategy configs
        final_config = self._merge_configs(self.master_config, self.strategy_config)
        
        # ARCHITECTURE FIX: Always build filters if config exists, regardless of master_filters_enabled
        # The master_filters_enabled flag is checked in the strategy's apply_filters call,
        # not here. This ensures filters are available for application.
        
        # Build calendar filters
        filters.extend(self._build_calendar_filters(final_config))
        
        # Build regime filters
        filters.extend(self._build_regime_filters(final_config))
        
        # Build news filters
        filters.extend(self._build_news_filters(final_config))
        
        # Build volume filters
        filters.extend(self._build_volume_filters(final_config))
        
        return filters
    
    def _merge_configs(self, master: Dict, strategy: Optional[StrategyConfig]) -> Dict:
        """
        Merge master config with strategy config.
        Strategy config overrides master config.
        
        Args:
            master: Master configuration dict
            strategy: Strategy configuration object
            
        Returns:
            Merged configuration dict
        """
        merged = master.copy()
        
        if strategy is None:
            return merged
        
        # Merge calendar filters
        if hasattr(strategy, 'calendar_filters'):
            merged.setdefault('calendar_filters', {})
            # Deep merge calendar filters
            self._deep_merge(merged['calendar_filters'], self._config_to_dict(strategy.calendar_filters))
        
        # Merge regime filters
        if hasattr(strategy, 'regime_filters'):
            merged.setdefault('regime_filters', {})
            # Deep merge regime filters
            self._deep_merge(merged['regime_filters'], self._config_to_dict(strategy.regime_filters))
        
        # Merge news filter
        if hasattr(strategy, 'news_filter'):
            merged.setdefault('news_filter', {})
            self._deep_merge(merged['news_filter'], self._config_to_dict(strategy.news_filter))
        
        # Merge volume filters
        if hasattr(strategy, 'volume_filters'):
            merged.setdefault('volume_filters', {})
            self._deep_merge(merged['volume_filters'], self._config_to_dict(strategy.volume_filters))
        
        return merged
    
    def _config_to_dict(self, config_obj) -> Dict:
        """Convert Pydantic config object to dict."""
        if hasattr(config_obj, 'model_dump'):
            # Only dump explicitly set fields so schema defaults don't
            # accidentally override master filter defaults.
            return config_obj.model_dump(exclude_unset=True)
        elif hasattr(config_obj, 'dict'):
            try:
                return config_obj.dict(exclude_unset=True)
            except TypeError:
                return config_obj.dict()
        elif isinstance(config_obj, dict):
            return config_obj
        else:
            return {}
    
    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override dict into base dict (in-place)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def apply_filters(
        self, 
        signal: pd.Series, 
        context: FilterContext
    ) -> FilterResult:
        """
        Apply all enabled filters to signal.
        
        Filters are applied in sequence. The first filter that fails causes
        the signal to be rejected (short-circuiting).
        
        Args:
            signal: Signal Series with direction, entry_price, etc.
            context: FilterContext with market data
            
        Returns:
            FilterResult indicating if signal passed all filters
        """
        # Update context with signal data
        context.signal_data = signal
        if 'direction' in signal:
            context.signal_direction = 1 if signal['direction'] == 'long' else -1
        
        # Ensure MarketSpec is available in context
        # Import MarketSpec here to avoid circular import at module level
        if context.market_spec is None:
            from engine.market import MarketSpec
            try:
                context.market_spec = MarketSpec.load_from_profiles(context.symbol)
            except (ValueError, FileNotFoundError):
                # Fallback: create basic MarketSpec
                context.market_spec = MarketSpec(
                    symbol=context.symbol,
                    exchange='unknown',
                    asset_class='crypto'  # Default fallback
                )
        
        # Apply filters in sequence (short-circuit on first failure)
        for filter_obj in self.filters:
            if not filter_obj.is_enabled():
                continue
            
            result = filter_obj.check(context)
            if not result.passed:
                # Track filter failures for debugging
                if not hasattr(self, '_filter_failure_counts'):
                    self._filter_failure_counts = {}
                filter_name = filter_obj.name
                count = self._filter_failure_counts.get(filter_name, 0)
                if count < 3:  # Print first 3 failures for each filter
                    print(f"[FilterManager] Filter {filter_name} rejected signal: {result.reason}")
                self._filter_failure_counts[filter_name] = count + 1
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Filter {filter_obj.name} rejected signal: {result.reason}")
                return result  # Short-circuit on first failure
        
        # All filters passed
        return FilterResult(passed=True)
    
    def _build_calendar_filters(self, config: Dict) -> List[FilterBase]:
        """Build calendar filter chain."""
        filters = []
        calendar_cfg = config.get('calendar_filters', {})
        
        # Day of week filter
        day_cfg = calendar_cfg.get('day_of_week', {}) or {}
        if day_cfg.get('enabled', False):
            from strategies.filters.calendar.day_of_week_filter import DayOfWeekFilter
            filters.append(DayOfWeekFilter(day_cfg))
        
        # Trading session filter (explicit opt-in)
        trading_sessions = calendar_cfg.get('trading_sessions')
        trading_sessions_enabled = bool(calendar_cfg.get('trading_sessions_enabled', False))

        # Legacy compatibility: if user configured allowed_sessions/instrument_sessions, treat that as enabling.
        if not trading_sessions_enabled:
            allowed_sessions = calendar_cfg.get('allowed_sessions', []) or []
            instrument_sessions = calendar_cfg.get('instrument_sessions', {}) or {}
            if (isinstance(allowed_sessions, list) and len(allowed_sessions) > 0) or (
                isinstance(instrument_sessions, dict) and len(instrument_sessions) > 0
            ):
                trading_sessions_enabled = True

        if trading_sessions_enabled and trading_sessions:
            from strategies.filters.calendar.trading_session_filter import TradingSessionFilter
            session_filter = TradingSessionFilter({'enabled': True, 'trading_sessions': trading_sessions})
            filters.append(session_filter)
        
        # Time blackout filter
        if calendar_cfg.get('time_blackouts_enabled', False):
            from strategies.filters.calendar.time_blackout_filter import TimeBlackoutFilter
            # Get time_blackouts config, which should have 'slots' dict
            time_blackouts_cfg = calendar_cfg.get('time_blackouts', {})
            # If it's a dict with 'slots', use it directly; otherwise wrap it
            if 'slots' not in time_blackouts_cfg and isinstance(time_blackouts_cfg, dict):
                # Assume the dict itself contains slot_1, slot_2, etc.
                time_blackouts_cfg = {'slots': time_blackouts_cfg}
            filters.append(TimeBlackoutFilter(time_blackouts_cfg))
        
        return filters
    
    def _build_regime_filters(self, config: Dict) -> List[FilterBase]:
        """Build regime filter chain."""
        filters = []
        regime_cfg = config.get('regime_filters', {}) or {}
        
        # ADX filter
        adx_cfg = regime_cfg.get('adx', {}) or {}
        if adx_cfg.get('enabled', False):
            from strategies.filters.regime.adx_filter import ADXFilter
            filters.append(ADXFilter(adx_cfg))
        
        # ATR Percentile filter (to be implemented)
        # if regime_cfg.get('atr_percentile', {}).get('enabled', False):
        #     from strategies.filters.regime import ATRPercentileFilter
        #     filters.append(ATRPercentileFilter(regime_cfg['atr_percentile']))
        
        # ATR Threshold filter
        atr_threshold_cfg = regime_cfg.get('atr_threshold', {}) or {}
        if atr_threshold_cfg.get('enabled', False):
            from strategies.filters.regime.atr_threshold_filter import ATRThresholdFilter
            filters.append(ATRThresholdFilter(atr_threshold_cfg))
        
        # SMA Distance filter
        sma_distance_cfg = regime_cfg.get('sma_distance', {}) or {}
        if sma_distance_cfg.get('enabled', False):
            from strategies.filters.regime.sma_distance_filter import SMADistanceFilter
            filters.append(SMADistanceFilter(sma_distance_cfg))
        
        # SMA Slope filter
        sma_slope_cfg = regime_cfg.get('sma_slope', {}) or {}
        if sma_slope_cfg.get('enabled', False):
            from strategies.filters.regime.sma_slope_filter import SMASlopeFilter
            filters.append(SMASlopeFilter(sma_slope_cfg))
        
        # EMA Expansion filter (to be implemented)
        # if regime_cfg.get('ema_expansion', {}).get('enabled', False):
        #     from strategies.filters.regime import EMAExpansionFilter
        #     filters.append(EMAExpansionFilter(regime_cfg['ema_expansion']))
        
        # Composite regime filter (to be implemented)
        # if regime_cfg.get('composite', {}).get('enabled', False):
        #     from strategies.filters.regime import CompositeRegimeFilter
        #     filters.append(CompositeRegimeFilter(regime_cfg['composite']))
        
        return filters
    
    def _build_news_filters(self, config: Dict) -> List[FilterBase]:
        """Build news filter chain."""
        filters = []
        news_cfg = config.get('news_filter', {}) or {}
        
        # News filter
        if news_cfg.get('enabled', False):
            from strategies.filters.news.news_filter import NewsFilter
            filters.append(NewsFilter(news_cfg))
        
        return filters
    
    def _build_volume_filters(self, config: Dict) -> List[FilterBase]:
        """Build volume filter chain."""
        filters = []
        volume_cfg = config.get('volume_filters', {}) or {}
        
        # Volume Increasing filter
        volume_increasing_cfg = volume_cfg.get('volume_increasing', {}) or {}
        if volume_increasing_cfg.get('enabled', False):
            from strategies.filters.volume.volume_increasing_filter import VolumeIncreasingFilter
            filters.append(VolumeIncreasingFilter(volume_increasing_cfg))
        
        return filters
