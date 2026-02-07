"""Day-of-week filter.

Allows trades only on configured weekdays.

- Monday = 0 ... Sunday = 6 (matches pandas Timestamp.weekday())
- All timestamps are assumed UTC+00 (consistent with other calendar filters)
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from strategies.filters.base import FilterBase, FilterContext, FilterResult


class DayOfWeekFilter(FilterBase):
    """Filter signals by allowed weekdays."""

    def __init__(self, config: Dict):
        super().__init__(config)
        allowed = config.get('allowed_days', [0, 1, 2, 3, 4])
        try:
            self.allowed_days: List[int] = [int(x) for x in allowed]
        except Exception:
            self.allowed_days = [0, 1, 2, 3, 4]

        # Keep only valid weekday values
        self.allowed_days = [d for d in self.allowed_days if 0 <= d <= 6]

    def check(self, context: FilterContext) -> FilterResult:
        if not self.enabled:
            return self._create_pass_result()

        if not isinstance(context.timestamp, pd.Timestamp):
            ts = pd.Timestamp(context.timestamp)
        else:
            ts = context.timestamp

        day_int = int(ts.weekday())  # 0=Monday ... 6=Sunday
        if day_int in self.allowed_days:
            return self._create_pass_result(metadata={'day': day_int, 'allowed_days': self.allowed_days})

        return self._create_fail_result(
            reason=f"Day-of-week blocked (day={day_int}, allowed={self.allowed_days})",
            metadata={'day': day_int, 'allowed_days': self.allowed_days},
        )
