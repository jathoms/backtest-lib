import datetime as dt

from backtest_lib.backtest.schedule import make_decision_schedule


def test_iterable_schedule_is_reiterable() -> None:
    schedule = make_decision_schedule(
        [dt.datetime(2025, 1, 1), dt.datetime(2025, 1, 2)]
    )
    assert list(schedule) == [
        dt.datetime(2025, 1, 1),
        dt.datetime(2025, 1, 2),
    ]
    assert list(schedule) == [
        dt.datetime(2025, 1, 1),
        dt.datetime(2025, 1, 2),
    ]


def test_interval_schedule_emits_steps() -> None:
    schedule = make_decision_schedule("2h", start=dt.datetime(2021, 11, 13))
    it = iter(schedule)
    assert next(it) == dt.datetime(2021, 11, 13, 0, 0)
    assert next(it) == dt.datetime(2021, 11, 13, 2, 0)


def test_cron_schedule_emits_hours() -> None:
    schedule = make_decision_schedule("0 * * * *", start=dt.datetime(2025, 2, 1))
    it = iter(schedule)
    assert next(it) == dt.datetime(2025, 2, 1, 1, 0)
    assert next(it) == dt.datetime(2025, 2, 1, 2, 0)


def test_bounded_schedule_stops_at_end() -> None:
    schedule = make_decision_schedule(
        "1w",
        start=dt.datetime(2024, 2, 1),
        end=dt.datetime(2024, 2, 3),
    )
    assert list(schedule) == [dt.datetime(2024, 2, 1, 0, 0)]
