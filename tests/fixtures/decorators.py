"""
Fixture: Functions and methods with decorators.
"""


def dummy_decorator(fn):
    return fn


def parameter_decorator(name: str):
    def wrapper(fn):
        return fn
    return wrapper


@dummy_decorator
@parameter_decorator("test")
def decorated_function(x: int) -> int:
    """A decorated standalone function."""
    return x + 1


class DecoratedClass:
    """Class with decorated methods."""

    @staticmethod
    def static_method(val: int) -> int:
        """A static utility method."""
        return val * 10

    @property
    def value(self) -> int:
        """A property getter."""
        return 42

    @dummy_decorator
    def multi_decorated_method(self, flag: bool) -> bool:
        """A decorated instance method."""
        return not flag
