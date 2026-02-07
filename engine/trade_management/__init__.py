"""Trade management module for abstracted exit logic."""

from engine.trade_management.manager import TradeManagementManager
from engine.trade_management.exit_resolver import ExitResolver, ExitCondition, ExitType

__all__ = ['TradeManagementManager', 'ExitResolver', 'ExitCondition', 'ExitType']

