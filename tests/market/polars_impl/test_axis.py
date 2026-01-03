from backtest_lib.market.polars_impl._axis import SecurityAxis


def test_static_constructor_security_axis():
    names = ["a", "b"]
    security_axis = SecurityAxis.from_names(names)
    assert security_axis.names == ("a", "b")
    assert security_axis.pos == {"a": 0, "b": 1}


def test_returning_length_of_names_security_axis():
    names = ["a", "b"]
    security_axis = SecurityAxis.from_names(names)
    assert len(security_axis.names) == 2


def test_ability_to_handle_empty_names_security_axis():
    security_axis = SecurityAxis.from_names([])
    assert len(security_axis.names) == 0
