import re
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timedelta
from typing import (
    overload,
)

from croniter import croniter
from dateutil.relativedelta import relativedelta

from backtest_lib.backtest._helpers import DateTimeLike, _to_pydt
from backtest_lib.market.timeseries import Comparable

_SECONDS_STRS = ("s", "sec", "secs", "second", "seconds")
_MINUTES_STRS = ("m", "min", "mins", "minute", "minutes")
_HOURS_STRS = ("h", "hr", "hrs", "hour", "hours")
_DAYS_STRS = ("d", "day", "days")
_WEEKS_STRS = ("w", "wk", "wks", "week", "weeks")
_MONTHS_STRS = ("mo", "mon", "mons", "month", "months")
_QUARTERS_STRS = ("q", "qtr", "qtrs", "quarter", "quarters")
_YEARS_STRS = ("y", "yr", "yrs", "year", "years")

_INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")
_PLAIN_INTERVAL_MAPPING = {
    "hourly": _HOURS_STRS[0],
    "daily": _DAYS_STRS[0],
    "weekly": _WEEKS_STRS[0],
    "monthly": _MONTHS_STRS[0],
    "quarterly": _QUARTERS_STRS[0],
    "yearly": _YEARS_STRS[0],
    "annually": _YEARS_STRS[0],
}


def _is_interval_string(s: str) -> bool:
    return _INTERVAL_RE.match(s) is not None or s in _PLAIN_INTERVAL_MAPPING


def _parse_step(s: str) -> timedelta | relativedelta:
    if s not in _PLAIN_INTERVAL_MAPPING:
        m = _INTERVAL_RE.match(s)
        if not m:
            raise ValueError(f"Invalid interval: {s!r}")

        n = int(m.group(1))
        unit = str(m.group(2)).lower()
        if n <= 0:
            raise ValueError("interval must be positive")
    else:
        n = 1
        unit = _PLAIN_INTERVAL_MAPPING[s]

    if unit in _SECONDS_STRS:
        return timedelta(seconds=n)
    if unit in _MINUTES_STRS:
        return timedelta(minutes=n)
    if unit in _HOURS_STRS:
        return timedelta(hours=n)
    if unit in _DAYS_STRS:
        return timedelta(days=n)
    if unit in _WEEKS_STRS:
        return timedelta(weeks=n)
    if unit in _MONTHS_STRS:
        return relativedelta(months=n)
    if unit in _QUARTERS_STRS:
        return relativedelta(months=3 * n)
    if unit in _YEARS_STRS:
        return relativedelta(years=n)

    raise ValueError(f"Unsupported interval unit: {unit!r}")


class DecisionSchedule[I: Comparable]:
    _schedule: Iterable[I]
    _start: I
    _end: I | None
    _inclusive_end: bool
    _schedule_str: str | None

    def __init__(
        self,
        schedule: Iterable[I],
        start: I,
        end: I | None = None,
        *,
        inclusive_end: bool = True,
        schedule_str: str | None = None,
    ) -> None:
        self._schedule = schedule
        self._start = start
        self._end = end
        self._inclusive_end = inclusive_end
        self._schedule_str = schedule_str

    @property
    def schedule(self) -> Iterable[I]:
        return self._schedule

    def __iter__(self) -> Iterator[I]:
        start = self._start
        end = self._end
        prev = None

        for i, x in enumerate(self._schedule):
            if prev is not None and x < prev:
                raise ValueError(
                    "Decision schedule is not non-decreasing, "
                    f"value {x} (idx:{i}) < {prev} (idx:{i - 1})"
                )
            if start is not None and x < start:
                continue
            if end is not None and ((not self._inclusive_end and x >= end) or x > end):
                break

            yield x


def _step_datetime_iter(
    step: timedelta | relativedelta, start: datetime
) -> Iterable[datetime]:
    def _it() -> Iterator[datetime]:
        cur = start
        while True:
            yield cur
            cur = cur + step

    return _IterFactoryIterable(_it)


def _cron_datetime_iter(cron: str, start: datetime) -> Iterable[datetime]:
    return _IterFactoryIterable(lambda: croniter(cron, start, datetime).all_next())


class _IterFactoryIterable[T]:
    """Wrap an iterator factory as an Iterable,
    so the inner schedule can always be Iterable[T].
    Each iteration produces a fresh iterator.
    """

    def __init__(self, factory: Callable[[], Iterator[T]]) -> None:
        self._factory = factory

    def __iter__(self) -> Iterator[T]:
        return self._factory()


def _raise_iterator_input_error(schedule: object) -> None:
    raise TypeError(
        "decision_schedule(...) requires a re-iterable schedule (e.g.,"
        f" list/tuple/range), not passed type {type(schedule)}. If you need to stream"
        " non-materialized values, use decision_schedule_factory(f) where `f` is a"
        " function that yields the values of your schedule "
    )


@overload
def make_decision_schedule(
    schedule: str,
    start: datetime,
    end: datetime | None = None,
    *,
    inclusive_end: bool = True,
) -> DecisionSchedule[datetime]: ...


@overload
def make_decision_schedule[I: Comparable](
    schedule: Iterable[I],
    start: I | None = None,
    end: I | None = None,
    *,
    inclusive_end: bool = True,
) -> DecisionSchedule[I]: ...


def make_decision_schedule[I: Comparable](
    schedule: Iterable[I] | str,
    start: I | datetime | None = None,
    end: I | datetime | None = None,
    *,
    inclusive_end: bool = True,
) -> DecisionSchedule:
    if isinstance(schedule, str):
        if not isinstance(start, DateTimeLike):
            raise TypeError("For string schedules, start must be a datetime.")
        if end is not None:
            if not isinstance(end, DateTimeLike):
                raise TypeError("For string schedules, end a datetime or None.")
            if end < start:
                raise ValueError("end must be >= start")
            end = _to_pydt(end)

        start = _to_pydt(start)

        s = schedule.strip()
        inferred = (
            _step_datetime_iter(_parse_step(s), start)
            if _is_interval_string(s)
            else _cron_datetime_iter(s, start)
        )
        return DecisionSchedule[datetime](
            inferred,
            start,
            end,
            inclusive_end=inclusive_end,
            schedule_str=s,
        )

    it = iter(schedule)
    if it is schedule:
        _raise_iterator_input_error(schedule)

    if start is None:
        # infer start from first element
        try:
            start = next(it)
        except StopIteration:
            raise ValueError("Cannot infer start from an empty schedule.") from None

    return DecisionSchedule(
        schedule,
        start,
        end,
        inclusive_end=inclusive_end,
    )


def decision_schedule_factory[I: Comparable](
    factory: Callable[[], Iterator[I]],
    start: I | None = None,
    end: I | None = None,
    *,
    inclusive_end: bool = True,
) -> DecisionSchedule[I]:
    schedule = _IterFactoryIterable(factory)

    if start is None:
        it = iter(schedule)
        try:
            start = next(it)
        except StopIteration:
            raise ValueError("Cannot infer start from an empty schedule.") from None

    assert start is not None

    return DecisionSchedule(
        schedule,
        start,
        end,
        inclusive_end=inclusive_end,
    )
