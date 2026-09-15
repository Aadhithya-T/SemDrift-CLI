"""
Fixture: Nested functions inside top-level functions and methods.
"""


def outer_function(val: int) -> int:
    """Outer container function."""

    def inner_helper(factor: int) -> int:
        """Inner helper calculation."""
        return val * factor

    return inner_helper(2)


class Container:
    """Class containing method with nested helper."""

    def compute(self, data: list[int]) -> int:
        """Compute aggregate value."""

        def reducer(acc: int, x: int) -> int:
            """Reducer helper."""
            return acc + x

        total = 0
        for item in data:
            total = reducer(total, item)
        return total
