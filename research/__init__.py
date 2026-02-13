"""Research-layer: human-readable strategy specs compiled into TradingLab strategies."""

from .spec import StrategySpec
from .compiler import StrategyCompiler
from .nl_parser import Clarification, ParseResult, parse_english_strategy

__all__ = ["StrategySpec", "StrategyCompiler", "Clarification", "ParseResult", "parse_english_strategy"]
