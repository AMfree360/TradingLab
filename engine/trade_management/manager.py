"""Trade management manager for merging master and strategy configs."""

from typing import Dict, Optional, Any
from pathlib import Path
import yaml
from config.schema import (
    StrategyConfig,
    StopLossConfig,
    TrailingStopConfig,
    PartialExitConfig,
    TakeProfitConfig,
    MinStopDistanceConfig
)


class TradeManagementManager:
    """Manages trade management configuration by merging master and strategy configs.
    
    Follows the same pattern as FilterManager:
    - Loads master config from config/master_trade_management.yml
    - Merges with strategy-specific config
    - Strategy config overrides master defaults
    """
    
    def __init__(
        self,
        master_config: Optional[Dict] = None,
        strategy_config: Optional[StrategyConfig] = None
    ):
        """Initialize trade management manager.
        
        Args:
            master_config: Master trade management configuration (from config/master_trade_management.yml)
            strategy_config: Strategy-specific configuration
        """
        self.master_config = master_config or {}
        self.strategy_config = strategy_config
        self.final_config: Optional[Dict] = None
        
        # Build final config if configs provided
        if self.master_config or self.strategy_config:
            self.final_config = self._merge_configs(self.master_config, self.strategy_config)
    
    @classmethod
    def from_strategy(cls, strategy_config: StrategyConfig) -> 'TradeManagementManager':
        """Create manager from strategy config, loading master config automatically.
        
        Args:
            strategy_config: Strategy configuration
            
        Returns:
            TradeManagementManager instance
        """
        # Load master config if available
        master_config = None
        master_config_path = Path(__file__).parent.parent.parent / 'config' / 'master_trade_management.yml'
        if master_config_path.exists():
            try:
                with open(master_config_path, 'r') as f:
                    master_data = yaml.safe_load(f)
                    master_config = master_data.get('master_trade_management', {})
            except Exception:
                # If master config can't be loaded, continue without it
                master_config = None
        
        return cls(master_config=master_config, strategy_config=strategy_config)
    
    def _merge_configs(self, master: Dict, strategy: Optional[StrategyConfig]) -> Dict:
        """Merge master and strategy configs.
        
        Strategy config overrides master defaults.
        
        Args:
            master: Master configuration dict
            strategy: Strategy configuration object
            
        Returns:
            Merged configuration dict
        """
        merged = master.copy() if master else {}
        
        if strategy:
            # Merge stop loss
            if hasattr(strategy, 'stop_loss') and strategy.stop_loss:
                merged['stop_loss'] = self._merge_stop_loss(
                    master.get('stop_loss', {}),
                    strategy.stop_loss
                )
            
            # Merge trailing stop
            if hasattr(strategy, 'trailing_stop') and strategy.trailing_stop:
                merged['trailing_stop'] = self._merge_trailing_stop(
                    master.get('trailing_stop', {}),
                    strategy.trailing_stop
                )
            
            # Merge take profit
            if hasattr(strategy, 'take_profit') and strategy.take_profit:
                merged['take_profit'] = self._merge_take_profit(
                    master.get('take_profit', {}),
                    strategy.take_profit
                )
            
            # Merge partial exit
            if hasattr(strategy, 'partial_exit') and strategy.partial_exit:
                merged['partial_exit'] = self._merge_partial_exit(
                    master.get('partial_exit', {}),
                    strategy.partial_exit
                )
        
        return merged
    
    def _merge_stop_loss(self, master: Dict, strategy: StopLossConfig) -> Dict:
        """Merge stop loss configs."""
        merged = master.copy() if master else {}
        
        # Convert strategy config to dict and merge
        strategy_dict = strategy.model_dump(exclude_none=True)
        merged.update(strategy_dict)
        
        # Handle min_stop_distance separately
        if hasattr(strategy, 'min_stop_distance') and strategy.min_stop_distance:
            master_min = master.get('min_stop_distance', {})
            strategy_min = strategy.min_stop_distance.model_dump(exclude_none=True)
            merged['min_stop_distance'] = {**master_min, **strategy_min}
        
        return merged
    
    def _merge_trailing_stop(self, master: Dict, strategy: TrailingStopConfig) -> Dict:
        """Merge trailing stop configs."""
        merged = master.copy() if master else {}
        strategy_dict = strategy.model_dump(exclude_none=True)
        merged.update(strategy_dict)
        return merged
    
    def _merge_take_profit(self, master: Dict, strategy: TakeProfitConfig) -> Dict:
        """Merge take profit configs."""
        merged = master.copy() if master else {}
        strategy_dict = strategy.model_dump(exclude_none=True)
        merged.update(strategy_dict)
        return merged
    
    def _merge_partial_exit(self, master: Dict, strategy: PartialExitConfig) -> Dict:
        """Merge partial exit configs."""
        merged = master.copy() if master else {}
        strategy_dict = strategy.model_dump(exclude_none=True)
        merged.update(strategy_dict)
        return merged
    
    def get_stop_loss_config(self) -> Dict:
        """Get merged stop loss configuration."""
        if not self.final_config:
            return {}
        return self.final_config.get('stop_loss', {})
    
    def get_trailing_stop_config(self) -> Dict:
        """Get merged trailing stop configuration."""
        if not self.final_config:
            return {}
        return self.final_config.get('trailing_stop', {})
    
    def get_take_profit_config(self) -> Dict:
        """Get merged take profit configuration."""
        if not self.final_config:
            return {}
        return self.final_config.get('take_profit', {})
    
    def get_partial_exit_config(self) -> Dict:
        """Get merged partial exit configuration."""
        if not self.final_config:
            return {}
        return self.final_config.get('partial_exit', {})

