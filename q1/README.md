# Q1: `enforce_dict_str_int` Decorator

A Python decorator that validates all arguments to a function are `dict[str, int]`.

## Usage

```python
from enforce_dict_str_int import enforce_dict_str_int

@enforce_dict_str_int
def process(data: dict[str, int]) -> int:
    return sum(data.values())

process({"a": 1, "b": 2})   # ✓ returns 3
process({"a": 1.5})          # ✗ TypeError: non-int value at key 'a'
process("not a dict")        # ✗ TypeError: must be dict[str, int], got str
```

Works with any function signature — positional, keyword, variadic, mixed:

```python
@enforce_dict_str_int
def merge(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {**a, **b}

merge({"a": 1}, b={"b": 2})  # ✓
```

## Run

```bash
# Self-test demo
python enforce_dict_str_int.py

# Full test suite (31 tests)
pip install pytest
pytest test_enforce_dict_str_int.py -v
```

## Design Decisions

**`bool` is rejected.** `isinstance(True, int)` is `True` in Python because `bool` is a subclass of `int`. But the spec says "whole number values" — booleans are semantically distinct, so the decorator explicitly excludes them. This prevents subtle bugs where `{"flag": True}` silently passes validation.

**Empty dicts are valid.** An empty `{}` satisfies `dict[str, int]` vacuously — there are no keys or values to violate the constraint. This matches Python's type system behaviour.

**`dict` subclasses are accepted.** `OrderedDict`, `defaultdict`, and other subclasses fulfill the `dict[str, int]` contract. Rejecting them would break the Liskov substitution principle and make the decorator less useful in real codebases.

**Validation happens before the function body.** If any argument fails, `TypeError` is raised before the decorated function executes. No side effects occur on invalid input.

**Error messages identify the failing argument.** The exception message includes the argument name (or position), the offending key/value, and the actual type — making debugging straightforward.
