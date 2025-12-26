from datetime import UTC, date, datetime
from typing import Any

import numpy as np

DateTimeLike = np.datetime64 | datetime | date


def _to_pydt(some_datetime: Any) -> datetime:
    if isinstance(some_datetime, datetime):
        return some_datetime
    elif isinstance(some_datetime, np.datetime64):
        if np.isnat(some_datetime):
            raise ValueError("Cannot convert NaT to Python datetime.")
        us = some_datetime.astype("datetime64[us]").astype(np.int64)
        return datetime.fromtimestamp(us / 1e6, UTC)
    else:
        raise TypeError(
            f"Cannot convert {some_datetime} with type {type(some_datetime)} to python datetime"
        )
