"""
dict_str_int_guard — a decorator that enforces dict[str, int] on all arguments.

Usage:
    @enforce_dict_str_int
    def process(data: dict[str, int]) -> int:
        return sum(data.values())

    process({"a": 1, "b": 2})   # ✓ returns 3
    process({"a": 1.5})          # ✗ TypeError
    process("not a dict")        # ✗ TypeError
    process({"a": True})         # ✗ TypeError (bool is not a whole number)
"""

from __future__ import annotations

import functools
from typing import Any, Callable


def _validate_dict_str_int(value: Any, name: str) -> None:
    """
    Validate that `value` is a dict with str keys and int values.

    Raises TypeError with a descriptive message on failure.

    Design decisions:
      - bool is rejected even though isinstance(True, int) is True,
        because the spec says "whole number values" and bools are
        semantically distinct from integers.
      - Empty dicts are valid — they satisfy dict[str, int] vacuously.
      - dict subclasses (OrderedDict, defaultdict) are accepted since
        they fulfill the dict[str, int] contract.
    """
    if not isinstance(value, dict):
        raise TypeError(
            f"Argument '{name}' must be dict[str, int], "
            f"got {type(value).__name__}"
        )

    for k, v in value.items():
        if not isinstance(k, str):
            raise TypeError(
                f"Argument '{name}' has non-str key: "
                f"{k!r} (type {type(k).__name__})"
            )
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(
                f"Argument '{name}' has non-int value at key {k!r}: "
                f"{v!r} (type {type(v).__name__})"
            )


def enforce_dict_str_int(fn: Callable) -> Callable:
    """
    Decorator that validates every argument to `fn` is a dict[str, int].

    Checks both positional and keyword arguments. Raises TypeError
    before the function body executes if any argument fails validation.

    Example:
        @enforce_dict_str_int
        def merge(*dicts: dict[str, int]) -> dict[str, int]:
            result = {}
            for d in dicts:
                result.update(d)
            return result
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for i, arg in enumerate(args):
            _validate_dict_str_int(arg, f"args[{i}]")
        for key, arg in kwargs.items():
            _validate_dict_str_int(arg, key)
        return fn(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Demo / self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    @enforce_dict_str_int
    def total(data: dict[str, int]) -> int:
        """Sum all values."""
        return sum(data.values())

    @enforce_dict_str_int
    def merge(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
        """Merge two dicts."""
        return {**a, **b}

    # --- Passing cases ---
    print("✓ total({'x': 1, 'y': 2})        =", total({"x": 1, "y": 2}))
    print("✓ total({})                       =", total({}))
    print("✓ merge({'a': 1}, b={'b': 2})     =", merge({"a": 1}, b={"b": 2}))

    # --- Failing cases ---
    cases = [
        ("non-dict",         lambda: total("hello")),
        ("float value",      lambda: total({"a": 1.5})),
        ("bool value",       lambda: total({"a": True})),
        ("non-str key",      lambda: total({1: 100})),
        ("None",             lambda: total(None)),
        ("nested dict",      lambda: total({"a": {"b": 1}})),
        ("mixed in kwarg",   lambda: merge({"a": 1}, b={"x": 3.14})),
    ]

    for label, call in cases:
        try:
            call()
            print(f"✗ {label}: should have raised TypeError")
        except TypeError as e:
            print(f"✓ {label}: caught → {e}")
