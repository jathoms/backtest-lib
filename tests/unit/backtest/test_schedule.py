import datetime as dt

import pytest

from backtest_lib.backtest.schedule import (
    DecisionSchedule,
    decision_schedule_factory,
    make_decision_schedule,
)


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


def test_plain_interval_string_expands_to_step_schedule() -> None:
    schedule = make_decision_schedule("daily", start=dt.datetime(2024, 3, 1))
    it = iter(schedule)
    assert next(it) == dt.datetime(2024, 3, 1, 0, 0)
    assert next(it) == dt.datetime(2024, 3, 2, 0, 0)


def test_interval_string_requires_datetime_start() -> None:
    with pytest.raises(TypeError):
        make_decision_schedule("1d", start="2024-01-01")


def test_interval_string_requires_datetime_end() -> None:
    with pytest.raises(TypeError):
        # type: ignore
        make_decision_schedule(
            "1d",
            start=dt.datetime(2024, 1, 1),
            end="2024-01-02",
        )


def test_interval_string_rejects_end_before_start() -> None:
    with pytest.raises(ValueError):
        make_decision_schedule(
            "1d",
            start=dt.datetime(2024, 1, 2),
            end=dt.datetime(2024, 1, 1),
        )


def test_interval_string_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError):
        make_decision_schedule("0h", start=dt.datetime(2024, 1, 1))


def test_interval_string_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError):
        make_decision_schedule("3lightyears", start=dt.datetime(2024, 1, 1))


def test_interval_string_rejects_invalid_format() -> None:
    with pytest.raises(ValueError):
        make_decision_schedule("nonsense", start=dt.datetime(2024, 1, 1))


def test_iterator_schedule_requires_reiterable_input() -> None:
    schedule_iter = iter([dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)])
    with pytest.raises(TypeError):
        make_decision_schedule(schedule_iter)


def test_infer_start_requires_non_empty_schedule() -> None:
    with pytest.raises(ValueError):
        make_decision_schedule([])


def test_schedule_iteration_enforces_non_decreasing_order() -> None:
    schedule = DecisionSchedule(
        [dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1)],
        start=dt.datetime(2024, 1, 1),
    )
    with pytest.raises(ValueError):
        list(schedule)


def test_schedule_inclusive_end_behavior() -> None:
    schedule = DecisionSchedule(
        [
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 1, 2),
            dt.datetime(2024, 1, 3),
        ],
        start=dt.datetime(2024, 1, 2),
        end=dt.datetime(2024, 1, 3),
        inclusive_end=False,
    )
    assert list(schedule) == [dt.datetime(2024, 1, 2)]


def test_decision_schedule_factory_infers_start() -> None:
    def factory():
        yield dt.datetime(2024, 1, 1)
        yield dt.datetime(2024, 1, 2)

    schedule = decision_schedule_factory(factory)
    assert list(schedule) == [
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 1, 2),
    ]


def test_decision_schedule_factory_requires_non_empty_iterable() -> None:
    def factory():
        if False:
            yield None

    with pytest.raises(ValueError):
        decision_schedule_factory(factory)
