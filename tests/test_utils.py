"""Tests for anypinn.lib.utils — find, find_or_raise."""

import pytest

from anypinn.lib.utils import find, find_or_raise


class TestFind:
    def test_finds_match(self):
        result = find([1, 2, 3, 4], lambda x: x > 2)
        assert result == 3

    def test_returns_default_on_no_match(self):
        result = find([1, 2, 3], lambda x: x > 10)
        assert result is None

    def test_custom_default(self):
        result = find([1, 2, 3], lambda x: x > 10, default=-1)
        assert result == -1

    def test_first_match_returned(self):
        result = find([5, 3, 7, 1], lambda x: x < 5)
        assert result == 3

    def test_empty_iterable(self):
        result = find([], lambda x: True)
        assert result is None


class TestFindOrRaise:
    def test_finds_match(self):
        result = find_or_raise([1, 2, 3], lambda x: x == 2)
        assert result == 2

    def test_raises_default_valueerror(self):
        with pytest.raises(ValueError, match="Element not found"):
            find_or_raise([1, 2], lambda x: x > 10)

    def test_raises_custom_exception_instance(self):
        with pytest.raises(KeyError, match="nope"):
            find_or_raise([1], lambda x: x > 10, KeyError("nope"))

    def test_raises_callable_exception(self):
        with pytest.raises(RuntimeError, match="custom"):
            find_or_raise([1], lambda x: x > 10, lambda: RuntimeError("custom"))
