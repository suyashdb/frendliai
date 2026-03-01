"""Tests for enforce_dict_str_int decorator."""

import pytest
from collections import OrderedDict, defaultdict
from enforce_dict_str_int import enforce_dict_str_int


# ---------------------------------------------------------------------------
# Fixtures: decorated functions with various signatures
# ---------------------------------------------------------------------------

@enforce_dict_str_int
def single_arg(data: dict[str, int]) -> int:
    return sum(data.values())


@enforce_dict_str_int
def multi_arg(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {**a, **b}


@enforce_dict_str_int
def kwargs_only(*, data: dict[str, int]) -> int:
    return len(data)


@enforce_dict_str_int
def variadic(*dicts: dict[str, int]) -> dict[str, int]:
    result = {}
    for d in dicts:
        result.update(d)
    return result


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

class TestValidInputs:
    def test_simple_dict(self):
        assert single_arg({"a": 1, "b": 2}) == 3

    def test_empty_dict(self):
        assert single_arg({}) == 0

    def test_negative_values(self):
        assert single_arg({"x": -10, "y": 5}) == -5

    def test_zero_value(self):
        assert single_arg({"z": 0}) == 0

    def test_large_int(self):
        assert single_arg({"big": 10**18}) == 10**18

    def test_multiple_positional(self):
        assert multi_arg({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_keyword_argument(self):
        assert kwargs_only(data={"x": 42}) == 1

    def test_mixed_positional_and_keyword(self):
        assert multi_arg({"a": 1}, b={"b": 2}) == {"a": 1, "b": 2}

    def test_variadic(self):
        assert variadic({"a": 1}, {"b": 2}, {"c": 3}) == {"a": 1, "b": 2, "c": 3}

    def test_variadic_empty(self):
        assert variadic() == {}

    def test_ordered_dict(self):
        """dict subclasses should be accepted."""
        assert single_arg(OrderedDict(a=1, b=2)) == 3

    def test_defaultdict(self):
        d = defaultdict(int, {"x": 5})
        assert single_arg(d) == 5


# ---------------------------------------------------------------------------
# Invalid inputs — type errors
# ---------------------------------------------------------------------------

class TestInvalidInputs:
    def test_string(self):
        with pytest.raises(TypeError, match="must be dict"):
            single_arg("hello")

    def test_list(self):
        with pytest.raises(TypeError, match="must be dict"):
            single_arg([1, 2, 3])

    def test_int(self):
        with pytest.raises(TypeError, match="must be dict"):
            single_arg(42)

    def test_none(self):
        with pytest.raises(TypeError, match="must be dict"):
            single_arg(None)

    def test_float_value(self):
        with pytest.raises(TypeError, match="non-int value"):
            single_arg({"a": 1.5})

    def test_string_value(self):
        with pytest.raises(TypeError, match="non-int value"):
            single_arg({"a": "oops"})

    def test_bool_value_true(self):
        """bool is a subclass of int but should be rejected (not a whole number)."""
        with pytest.raises(TypeError, match="non-int value"):
            single_arg({"a": True})

    def test_bool_value_false(self):
        with pytest.raises(TypeError, match="non-int value"):
            single_arg({"a": False})

    def test_none_value(self):
        with pytest.raises(TypeError, match="non-int value"):
            single_arg({"a": None})

    def test_nested_dict_value(self):
        with pytest.raises(TypeError, match="non-int value"):
            single_arg({"a": {"nested": 1}})

    def test_non_str_key_int(self):
        with pytest.raises(TypeError, match="non-str key"):
            single_arg({1: 100})

    def test_non_str_key_tuple(self):
        with pytest.raises(TypeError, match="non-str key"):
            single_arg({("a",): 1})

    def test_second_arg_invalid(self):
        """First arg valid, second arg invalid — should fail on second."""
        with pytest.raises(TypeError, match="args\\[1\\]"):
            multi_arg({"a": 1}, {"b": 2.5})

    def test_kwarg_invalid(self):
        with pytest.raises(TypeError, match="data"):
            kwargs_only(data={"a": True})

    def test_variadic_one_bad(self):
        with pytest.raises(TypeError):
            variadic({"a": 1}, "not a dict", {"c": 3})


# ---------------------------------------------------------------------------
# Decorator preserves function metadata
# ---------------------------------------------------------------------------

class TestDecoratorBehaviour:
    def test_preserves_name(self):
        assert single_arg.__name__ == "single_arg"

    def test_preserves_docstring(self):
        @enforce_dict_str_int
        def documented(d: dict[str, int]) -> int:
            """This is documented."""
            return 0

        assert documented.__doc__ == "This is documented."

    def test_preserves_return_value(self):
        @enforce_dict_str_int
        def identity(d: dict[str, int]) -> dict[str, int]:
            return d

        data = {"a": 1, "b": 2}
        assert identity(data) == data

    def test_error_before_body_executes(self):
        """Validation should happen before the function body runs."""
        side_effects = []

        @enforce_dict_str_int
        def tracked(d: dict[str, int]) -> None:
            side_effects.append("executed")

        with pytest.raises(TypeError):
            tracked("bad input")

        assert side_effects == [], "Function body should not have executed"
