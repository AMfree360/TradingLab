"""Main benchmark runner.

Orchestrates trade import, Trading Lab backtest, and comparison.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd

from engine.backtest_engine import BacktestEngine, BacktestResult
from engine.market import MarketSpec
from strategies.base import StrategyBase
from config.schema import load_and_validate_strategy_config, load_config, validate_strategy_config
from config.market_loader import apply_market_profile

from .importers import import_trades_from_file
from .comparison import TradeComparator, MetricComparator, ComparisonResult
import importlib.util
import sys


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    comparison_result: ComparisonResult
    trading_lab_result: BacktestResult
    external_trades: List
    platform_name: str
    strategy_name: str
    data_file: str
    is_equivalent: bool
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        return {
            'platform': self.platform_name,
            'strategy': self.strategy_name,
            'is_equivalent': self.is_equivalent,
            'trade_match_rate': self.comparison_result.trade_comparison.match_rate,
            'metric_match': self.comparison_result.metric_comparison.overall_match,
            'trading_lab_trades': self.comparison_result.trade_comparison.total_trading_lab_trades,
            'external_trades': self.comparison_result.trade_comparison.total_external_trades,
            'matched_trades': self.comparison_result.trade_comparison.matched_trades,
        }


class BenchmarkRunner:
    """Main benchmark runner for comparing Trading Lab with external platforms."""
    
    def __init__(
        self,
        time_tolerance_seconds: int = 60,
        price_tolerance_pct: float = 0.01,
        metric_tolerance_pct: float = 5.0
    ):
        """
        Initialize benchmark runner.
        
        Args:
            time_tolerance_seconds: Time tolerance for trade matching
            price_tolerance_pct: Price tolerance for trade matching
            metric_tolerance_pct: Metric comparison tolerance percentage
        """
        self.trade_comparator = TradeComparator(
            time_tolerance_seconds=time_tolerance_seconds,
            price_tolerance_pct=price_tolerance_pct,
        )
        self.metric_comparator = MetricComparator(
            tolerance_pct=metric_tolerance_pct
        )
    
    def run_benchmark(
        self,
        strategy_name: str,
        data_file: str,
        external_trades_file: str,
        platform: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.0,
        market_overrides: Optional[Dict] = None
    ) -> BenchmarkResult:
        """Run benchmark comparison.
        
        Args:
            strategy_name: Name of strategy to test
            data_file: Path to market data file
            external_trades_file: Path to external platform trade export
            platform: Platform name ('mt5', 'ninjatrader', 'tradingview')
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital
            risk_free_rate: Risk-free rate for metrics
            market_overrides: Market configuration overrides
            
        Returns:
            BenchmarkResult
        """
        # Find strategy directory
        project_root = Path(__file__).parent.parent.parent
        strategy_dir = project_root / 'strategies' / strategy_name
        if not strategy_dir.exists():
            raise FileNotFoundError(f"Strategy '{strategy_name}' not found in strategies/")
        
        # Load config
        config_path = strategy_dir / 'config.yml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config as dict first to apply market profile
        config_dict = load_config(config_path)
        
        # Detect symbol from data file name for market profile
        symbol = Path(data_file).stem.split('_')[0]
        
        # Apply market profile automatically if symbol is in market profiles
        if symbol:
            try:
                config_dict = apply_market_profile(config_dict, symbol=symbol)
            except Exception as e:
                # If market profile not found, continue with defaults
                pass
        
        # Validate and create config object from updated dict
        strategy_config = validate_strategy_config(config_dict)
        
        # Load strategy class dynamically
        strategy_module_path = strategy_dir / 'strategy.py'
        if not strategy_module_path.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_module_path}")
        
        spec = importlib.util.spec_from_file_location(
            f"strategies.{strategy_name}.strategy",
            strategy_module_path
        )
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        
        # Find strategy class (assume it's the only class that inherits from StrategyBase)
        strategy_class = None
        for name in dir(strategy_module):
            obj = getattr(strategy_module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, StrategyBase) and 
                obj != StrategyBase):
                strategy_class = obj
                break
        
        if strategy_class is None:
            raise ValueError(f"Could not find strategy class in {strategy_module_path}")
        
        # Create strategy instance
        strategy = strategy_class(strategy_config)
        
        # Load data
        data = self._load_data(data_file, start_date, end_date)
        
        # Extract symbol from data if available, otherwise use detected symbol
        if 'symbol' in data.columns:
            symbol = data['symbol'].iloc[0]
        
        # Create market spec from profile
        try:
            market_spec = MarketSpec.load_from_profiles(symbol)
        except (ValueError, FileNotFoundError):
            # Fallback: create basic MarketSpec from config
            market_config = strategy_config.market
            asset_class = 'futures' if 'MES' in symbol or 'ES' in symbol else 'forex'
            market_spec = MarketSpec(
                symbol=symbol,
                exchange=getattr(market_config, 'exchange', 'unknown'),
                asset_class=asset_class,
                market_type=getattr(market_config, 'market_type', 'futures' if asset_class == 'futures' else 'spot'),
                leverage=getattr(market_config, 'leverage', 1.0),
                commission_rate=getattr(strategy_config.backtest, 'commissions', 0.0004),
                slippage_ticks=getattr(strategy_config.backtest, 'slippage_ticks', 0.0)
            )
        
        # Run Trading Lab backtest
        engine = BacktestEngine(
            strategy=strategy,
            market_spec=market_spec,
            initial_capital=initial_capital
        )
        
        trading_lab_result = engine.run(data)
        
        # Import external trades
        external_trades = import_trades_from_file(external_trades_file, platform)
        
        # Compare trades
        trade_comparison = self.trade_comparator.compare_trades(
            trading_lab_trades=trading_lab_result.trades,
            external_trades=external_trades
        )
        
        # Extract external metrics (if available in file or calculate from trades)
        external_metrics = self._extract_external_metrics(external_trades, trading_lab_result)
        
        # Compare metrics
        metric_comparison = self.metric_comparator.compare_metrics(
            trading_lab_result=trading_lab_result,
            external_metrics=external_metrics,
            risk_free_rate=risk_free_rate
        )
        
        # Create comparison result
        comparison_result = ComparisonResult(
            trade_comparison=trade_comparison,
            metric_comparison=metric_comparison,
            platform_name=platform,
            strategy_name=strategy_name,
            data_period=(
                trading_lab_result.trades[0].entry_time if trading_lab_result.trades else pd.Timestamp.now(),
                trading_lab_result.trades[-1].exit_time if trading_lab_result.trades else pd.Timestamp.now()
            )
        )
        
        return BenchmarkResult(
            comparison_result=comparison_result,
            trading_lab_result=trading_lab_result,
            external_trades=external_trades,
            platform_name=platform,
            strategy_name=strategy_name,
            data_file=data_file,
            is_equivalent=comparison_result.is_equivalent()
        )
    
    def _load_data(
        self,
        data_file: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load and filter market data."""
        path = Path(data_file)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Try to use DataLoader first (handles various formats better)
        try:
            from adapters.data.data_loader import DataLoader
            loader = DataLoader()
            data = loader.load(path)
        except (ImportError, Exception) as e:
            # Fallback to manual parsing for non-standard formats
            if path.suffix.lower() == '.csv':
                # First, try to read a sample to understand the format
                with open(data_file, 'r') as f:
                    first_line = f.readline().strip()
                
                # Check if the entire line is in format: "YYYYMMDD HHMMSS;O;H;L;C;V"
                if ';' in first_line and len(first_line.split(';')) >= 6:
                    # Parse line-by-line for this specific format
                    def parse_row(row_str):
                        """Parse a row like '20250914 004800;6645.25;6645.25;6645.25;6645.25;1'"""
                        parts = str(row_str).strip().split(';')
                        if len(parts) >= 6:
                            timestamp_str = parts[0].strip()
                            # Parse YYYYMMDD HHMMSS format
                            if len(timestamp_str) >= 14:
                                date_part = timestamp_str[:8]  # YYYYMMDD
                                time_part = timestamp_str[9:15] if len(timestamp_str) > 9 else "000000"  # HHMMSS
                                datetime_str = f"{date_part} {time_part}"
                                try:
                                    dt = pd.to_datetime(datetime_str, format='%Y%m%d %H%M%S')
                                except:
                                    dt = pd.to_datetime(datetime_str)
                                return {
                                    'timestamp': dt,
                                    'open': float(parts[1]),
                                    'high': float(parts[2]),
                                    'low': float(parts[3]),
                                    'close': float(parts[4]),
                                    'volume': float(parts[5]) if len(parts) > 5 else 0
                                }
                        return None
                    
                    # Read and parse all lines
                    parsed_rows = []
                    with open(data_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:  # Skip empty lines
                                parsed = parse_row(line)
                                if parsed:
                                    parsed_rows.append(parsed)
                    
                    if parsed_rows:
                        data = pd.DataFrame(parsed_rows)
                        data.set_index('timestamp', inplace=True)
                    else:
                        raise ValueError(f"Could not parse timestamp format in {data_file}")
                else:
                    # Try different separators for standard CSV
                    separators = [';', ',', '\t']
                    data = None
                    for sep in separators:
                        try:
                            # Try reading with header
                            data = pd.read_csv(data_file, sep=sep, header=0)
                            # Check if we got reasonable columns
                            if len(data.columns) > 1:
                                break
                        except:
                            continue
                    
                    # If header reading failed, try without header
                    if data is None or len(data.columns) == 1:
                        for sep in separators:
                            try:
                                data = pd.read_csv(data_file, sep=sep, header=None)
                                # Check if first row looks like data (not header)
                                if len(data.columns) >= 5:  # At least timestamp + OHLC
                                    break
                            except:
                                continue
                    
                    if data is None:
                        raise ValueError(f"Could not parse CSV file: {data_file}")
                    
                    # Handle non-standard formats
                    # Check if first column contains timestamp-like data
                    first_col = data.columns[0]
                    first_val = str(data.iloc[0, 0]) if len(data) > 0 else ""
                    
                    # Check for format like "20250914 004800" (YYYYMMDD HHMMSS)
                    if len(first_val) >= 14 and any(c.isdigit() for c in first_val[:14]):
                        # Try to parse first column as datetime
                        try:
                            # Handle YYYYMMDD HHMMSS format
                            def parse_timestamp(ts_str):
                                ts_str = str(ts_str).strip()
                                if len(ts_str) >= 14:
                                    date_part = ts_str[:8]  # YYYYMMDD
                                    time_part = ts_str[9:15] if len(ts_str) > 9 else "000000"  # HHMMSS
                                    datetime_str = f"{date_part} {time_part}"
                                    return pd.to_datetime(datetime_str, format='%Y%m%d %H%M%S')
                                return pd.to_datetime(ts_str)
                            
                            data[first_col] = data[first_col].apply(parse_timestamp)
                            data.set_index(first_col, inplace=True)
                        except (ValueError, TypeError) as e:
                            # If that fails, try standard datetime parsing
                            try:
                                data[first_col] = pd.to_datetime(data[first_col])
                                data.set_index(first_col, inplace=True)
                            except:
                                raise ValueError(
                                    f"Could not parse timestamp column '{first_col}' in {data_file}: {e}"
                                )
                    else:
                        # Standard CSV format - try to find timestamp column
                        timestamp_cols = ['timestamp', 'datetime', 'date', 'time', 'Timestamp', 'DateTime', 'Date', 'Time']
                        timestamp_col = None
                        for col in timestamp_cols:
                            if col in data.columns:
                                timestamp_col = col
                                break
                        
                        if timestamp_col:
                            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
                            data.set_index(timestamp_col, inplace=True)
                        elif isinstance(data.index, pd.RangeIndex) and len(data.columns) > 0:
                            # Try first column as datetime
                            first_col = data.columns[0]
                            try:
                                data[first_col] = pd.to_datetime(data[first_col])
                                data.set_index(first_col, inplace=True)
                            except (ValueError, TypeError):
                                raise ValueError(
                                    f"Could not determine datetime index for {data_file}. "
                                    f"Expected timestamp column or datetime index. "
                                    f"Columns: {data.columns.tolist()}"
                                )
                        else:
                            # Try to convert existing index
                            try:
                                data.index = pd.to_datetime(data.index)
                            except (ValueError, TypeError):
                                raise ValueError(
                                    f"Could not convert index to datetime for {data_file}. "
                                    f"Index type: {type(data.index)}, dtype: {data.index.dtype}"
                                )
            elif path.suffix.lower() == '.parquet':
                data = pd.read_parquet(data_file)
                # Ensure datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    if 'timestamp' in data.columns:
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data.set_index('timestamp', inplace=True)
                    else:
                        data.index = pd.to_datetime(data.index)
            else:
                raise ValueError(f"Unsupported data format: {path.suffix}")
        
        # Ensure we have a DatetimeIndex before filtering
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"Data index is not DatetimeIndex after loading: {type(data.index)}")
        
        # Filter by date range
        if start_date:
            start_ts = pd.Timestamp(start_date)
            data = data[data.index >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date)
            data = data[data.index <= end_ts]
        
        return data
    
    def _extract_external_metrics(
        self,
        external_trades: List,
        trading_lab_result: BacktestResult
    ) -> Dict[str, float]:
        """Extract or calculate metrics from external trades.
        
        For now, we calculate basic metrics from the external trades.
        In the future, this could read metrics directly from the export file if available.
        """
        if not external_trades:
            return {}
        
        metrics = {}
        
        # Basic metrics
        total_trades = len(external_trades)
        winning_trades = [t for t in external_trades if hasattr(t, 'pnl_after_costs') and t.pnl_after_costs > 0]
        losing_trades = [t for t in external_trades if hasattr(t, 'pnl_after_costs') and t.pnl_after_costs < 0]
        
        metrics['total_trades'] = float(total_trades)
        metrics['winning_trades'] = float(len(winning_trades))
        metrics['losing_trades'] = float(len(losing_trades))
        
        # Net profit
        net_profit = sum(t.pnl_after_costs for t in external_trades if hasattr(t, 'pnl_after_costs'))
        metrics['net_profit'] = net_profit
        
        # Win rate
        if total_trades > 0:
            metrics['win_rate'] = (len(winning_trades) / total_trades) * 100.0
        else:
            metrics['win_rate'] = 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl_after_costs for t in winning_trades)
        gross_loss = abs(sum(t.pnl_after_costs for t in losing_trades))
        if gross_loss > 0:
            metrics['profit_factor'] = gross_profit / gross_loss
        else:
            metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
        
        # Average win/loss
        if winning_trades:
            metrics['avg_win'] = sum(t.pnl_after_costs for t in winning_trades) / len(winning_trades)
        else:
            metrics['avg_win'] = 0.0
        
        if losing_trades:
            metrics['avg_loss'] = sum(t.pnl_after_costs for t in losing_trades) / len(losing_trades)
        else:
            metrics['avg_loss'] = 0.0
        
        # Expectancy
        if total_trades > 0:
            metrics['expectancy'] = net_profit / total_trades
        else:
            metrics['expectancy'] = 0.0
        
        # Max drawdown (simplified - would need equity curve)
        # For now, use Trading Lab's max drawdown as reference
        metrics['max_drawdown'] = trading_lab_result.max_drawdown
        
        # Sharpe/Sortino (would need returns series - skip for now)
        metrics['sharpe_ratio'] = 0.0
        metrics['sortino_ratio'] = 0.0
        metrics['recovery_factor'] = 0.0
        
        return metrics
