"""Helpers for working with instrument-native distance units.

These are used for UI labeling (pips/ticks/price) and for converting between
"native" distances and absolute price distances.

Design goals:
- Keep conversions deterministic.
- Prefer instrument-native primary units:
  - Forex: pips
  - Futures (with tick_size): ticks
  - Otherwise: price

NOTE: This module intentionally does not depend on GUI code.
"""

from __future__ import annotations

from typing import Literal

from engine.market import MarketSpec


NativeDistanceUnit = Literal["pips", "ticks", "price"]


def native_distance_unit(market_spec: MarketSpec) -> NativeDistanceUnit:
    if getattr(market_spec, "asset_class", None) == "forex":
        return "pips"

    tick_size = getattr(market_spec, "tick_size", None)
    if tick_size is not None and float(tick_size) > 0:
        # For now, prefer ticks for any instrument with an explicit tick_size.
        # (Later we can refine this to prefer `price` for crypto/stocks and `ticks` for futures.)
        if getattr(market_spec, "asset_class", None) == "futures":
            return "ticks"

    return "price"


def native_to_price_distance(market_spec: MarketSpec, value_native: float) -> float:
    """Convert a native distance (pips/ticks/price) into a price distance."""
    unit = native_distance_unit(market_spec)

    if unit == "pips":
        pip_value = float(getattr(market_spec, "pip_value", None) or 0.0001)
        return float(value_native) * pip_value

    if unit == "ticks":
        tick_size = float(getattr(market_spec, "tick_size", None) or 0.0)
        if tick_size <= 0:
            raise ValueError("tick_size is required for ticks conversion")
        return float(value_native) * tick_size

    return float(value_native)


def price_to_native_distance(market_spec: MarketSpec, value_price: float) -> float:
    """Convert a price distance into native distance units (pips/ticks/price)."""
    unit = native_distance_unit(market_spec)

    if unit == "pips":
        pip_value = float(getattr(market_spec, "pip_value", None) or 0.0001)
        if pip_value <= 0:
            raise ValueError("pip_value must be > 0 for pips conversion")
        return float(value_price) / pip_value

    if unit == "ticks":
        tick_size = float(getattr(market_spec, "tick_size", None) or 0.0)
        if tick_size <= 0:
            raise ValueError("tick_size is required for ticks conversion")
        return float(value_price) / tick_size

    return float(value_price)
