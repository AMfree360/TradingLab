"""Deprecated compatibility module.

`engine.market` is the single source of truth for `MarketSpec`.

This file exists only so older code that imports from `engine.market_spec`
keeps working.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from engine.market import MarketSpec as _MarketSpec
from engine.broker import BrokerModel as BrokerModel


class MarketSpec(_MarketSpec):
    """Compatibility wrapper around `engine.market.MarketSpec`.

    Provides the legacy `from_market_profile` constructor used by older docs.
    """

    @classmethod
    def from_market_profile(
        cls,
        symbol: str,
        profile: Dict[str, Any],
        asset_class_defaults: Optional[Dict[str, Any]] = None,
    ) -> "MarketSpec":
        merged: Dict[str, Any] = {}
        if asset_class_defaults:
            merged.update(asset_class_defaults)
        merged.update(profile)

        return cls(
            symbol=merged.get("symbol", symbol),
            exchange=merged.get("exchange", "unknown"),
            asset_class=merged.get("asset_class", "crypto"),
            market_type=merged.get("market_type", "spot"),
            leverage=merged.get("leverage", 1.0),
            contract_size=merged.get("contract_size"),
            pip_value=merged.get("pip_value"),
            tick_value=merged.get("tick_value"),
            min_trade_size=merged.get("min_trade_size", 0.01),
            lot_step=merged.get("lot_step", 0.01),
            price_precision=merged.get("price_precision", 5),
            quantity_precision=merged.get("quantity_precision", 2),
            commission_rate=merged.get("commission_rate", 0.0004),
            commission_per_contract=merged.get("commission_per_contract"),
            slippage_ticks=merged.get("slippage_ticks", 0.0),
            initial_margin_per_contract=merged.get("initial_margin_per_contract"),
            intraday_margin_per_contract=merged.get("intraday_margin_per_contract"),
            maintenance_margin_rate=merged.get("maintenance_margin_rate"),
            margin_mode=merged.get("margin_mode", "cross"),
        )

    @classmethod
    def load_from_profiles(cls, symbol: str, profiles_path=None) -> "MarketSpec":
        spec = _MarketSpec.load_from_profiles(symbol=symbol, profiles_path=profiles_path)
        return cls(**spec.__dict__)


# Legacy alias: older code may import `Broker` from here.
Broker = BrokerModel

__all__ = ["MarketSpec", "BrokerModel", "Broker"]

