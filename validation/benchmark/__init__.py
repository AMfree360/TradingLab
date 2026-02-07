"""Benchmarking module for comparing Trading Lab against external platforms.

This module provides tools to:
1. Import trades from external platforms (MT5, NinjaTrader, TradingView)
2. Compare trades trade-by-trade (engine equivalence)
3. Compare metrics (metric parity)
4. Generate comparison reports

Focus: Engine correctness and metric accuracy, not performance comparison.
"""

from .importers import (
    TradeImporter,
    MT5Importer,
    NinjaTraderImporter,
    TradingViewImporter,
    import_trades_from_file,
)

from .comparison import (
    TradeComparison,
    MetricComparison,
    ComparisonResult,
)

from .benchmark import (
    BenchmarkRunner,
    BenchmarkResult,
)

__all__ = [
    'TradeImporter',
    'MT5Importer',
    'NinjaTraderImporter',
    'TradingViewImporter',
    'import_trades_from_file',
    'TradeComparison',
    'MetricComparison',
    'ComparisonResult',
    'BenchmarkRunner',
    'BenchmarkResult',
]
