"""
Unit tests for semdrift.parser.ast_parser and semdrift.parser.models.
"""

from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from semdrift.parser.ast_parser import PythonASTParser
from semdrift.parser.models import CodeDocumentPair, ParseError


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


class TestCodeDocumentPairModel:
    """Verify CodeDocumentPair immutability and attributes."""

    def test_immutability(self):
        pair = CodeDocumentPair(
            file_path="foo.py",
            qualified_name="foo",
            line_number=1,
            code="def foo(): pass",
            docstring=None,
        )
        with pytest.raises(FrozenInstanceError):
            pair.qualified_name = "modified"  # type: ignore

    def test_default_values(self):
        pair = CodeDocumentPair(
            file_path="foo.py",
            qualified_name="foo",
            line_number=1,
            code="def foo(): pass",
        )
        assert pair.docstring is None
        assert pair.end_line_number is None
        assert pair.is_method is False
        assert pair.is_async is False
        assert pair.class_name is None


class TestPythonASTParser:
    """Verify PythonASTParser behavior against source strings and fixture files."""

    @pytest.fixture
    def parser(self):
        return PythonASTParser()

    # ------------------------------------------------------------------
    # Top-Level Functions
    # ------------------------------------------------------------------

    def test_top_level_function_with_docstring(self, parser):
        file_path = FIXTURES_DIR / "simple_function.py"
        pairs = parser.parse_file(file_path)

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.qualified_name == "calculate"
        assert pair.is_method is False
        assert pair.is_async is False
        assert pair.class_name is None
        assert pair.docstring == "Calculate the double of x."
        assert pair.line_number == 6
        assert pair.end_line_number == 8
        assert "def calculate(x: int) -> int:" in pair.code
        assert '"""Calculate the double of x."""' in pair.code
        assert "return x * 2" in pair.code

    def test_top_level_function_without_docstring(self, parser):
        file_path = FIXTURES_DIR / "undocumented.py"
        pairs = parser.parse_file(file_path)

        assert len(pairs) == 1
        pair = pairs[0]
        assert pair.qualified_name == "undocumented_sum"
        assert pair.docstring is None
        assert pair.is_method is False
        assert "return a + b" in pair.code

    # ------------------------------------------------------------------
    # Class Methods
    # ------------------------------------------------------------------

    def test_class_methods(self, parser):
        file_path = FIXTURES_DIR / "class_methods.py"
        pairs = parser.parse_file(file_path)

        assert len(pairs) == 2

        add_method = next(p for p in pairs if p.qualified_name == "Calculator.add")
        assert add_method.is_method is True
        assert add_method.is_async is False
        assert add_method.class_name == "Calculator"
        assert add_method.docstring == "Add two numbers together."
        assert "    def add(self, a: int, b: int) -> int:" in add_method.code

        sub_method = next(p for p in pairs if p.qualified_name == "Calculator.subtract")
        assert sub_method.is_method is True
        assert sub_method.class_name == "Calculator"
        assert sub_method.docstring is None

    # ------------------------------------------------------------------
    # Async Functions and Methods
    # ------------------------------------------------------------------

    def test_async_function_and_method(self, parser):
        file_path = FIXTURES_DIR / "async_defs.py"
        pairs = parser.parse_file(file_path)

        assert len(pairs) == 2

        async_func = next(p for p in pairs if p.qualified_name == "fetch_data")
        assert async_func.is_async is True
        assert async_func.is_method is False
        assert async_func.class_name is None
        assert async_func.docstring == "Fetch remote payload asynchronously."

        async_method = next(p for p in pairs if p.qualified_name == "AsyncService.execute_task")
        assert async_method.is_async is True
        assert async_method.is_method is True
        assert async_method.class_name == "AsyncService"
        assert async_method.docstring == "Execute task asynchronously."

    # ------------------------------------------------------------------
    # Nested Functions
    # ------------------------------------------------------------------

    def test_nested_functions(self, parser):
        file_path = FIXTURES_DIR / "nested_functions.py"
        pairs = parser.parse_file(file_path)

        names = [p.qualified_name for p in pairs]
        assert "outer_function" in names
        assert "outer_function.inner_helper" in names
        assert "Container.compute" in names
        assert "Container.compute.reducer" in names

        outer = next(p for p in pairs if p.qualified_name == "outer_function")
        assert outer.is_method is False
        assert outer.class_name is None

        inner = next(p for p in pairs if p.qualified_name == "outer_function.inner_helper")
        assert inner.is_method is False
        assert inner.class_name is None
        assert inner.docstring == "Inner helper calculation."

        comp = next(p for p in pairs if p.qualified_name == "Container.compute")
        assert comp.is_method is True
        assert comp.class_name == "Container"

        reducer = next(p for p in pairs if p.qualified_name == "Container.compute.reducer")
        assert reducer.is_method is False
        assert reducer.class_name is None
        assert reducer.docstring == "Reducer helper."

    # ------------------------------------------------------------------
    # Decorators & Indentation Preservation
    # ------------------------------------------------------------------

    def test_decorators_and_indentation(self, parser):
        file_path = FIXTURES_DIR / "decorators.py"
        pairs = parser.parse_file(file_path)

        func = next(p for p in pairs if p.qualified_name == "decorated_function")
        assert "@dummy_decorator" in func.code
        assert '@parameter_decorator("test")' in func.code
        assert "def decorated_function(x: int) -> int:" in func.code
        # First decorator starts the code snippet
        first_line = func.code.splitlines()[0]
        assert first_line == "@dummy_decorator"

        static_m = next(p for p in pairs if p.qualified_name == "DecoratedClass.static_method")
        assert static_m.is_method is True
        # Indentation relative to class should be preserved (starts with 4 spaces)
        lines = static_m.code.splitlines()
        assert lines[0] == "    @staticmethod"
        assert lines[1] == "    def static_method(val: int) -> int:"
        assert "        return val * 10" in lines

        prop = next(p for p in pairs if p.qualified_name == "DecoratedClass.value")
        assert prop.is_method is True
        assert "    @property" in prop.code

        multi = next(p for p in pairs if p.qualified_name == "DecoratedClass.multi_decorated_method")
        assert "    @dummy_decorator" in multi.code
        assert "    def multi_decorated_method" in multi.code

    # ------------------------------------------------------------------
    # Multiline Docstrings
    # ------------------------------------------------------------------

    def test_multiline_docstring(self, parser):
        file_path = FIXTURES_DIR / "multiline_docstring.py"
        pairs = parser.parse_file(file_path)

        assert len(pairs) == 1
        doc = pairs[0].docstring
        assert doc is not None
        assert doc.startswith("Format an address string cleanly.")
        assert "Parameters:" in doc
        assert "street: The street name and number." in doc
        assert '"""' not in doc

    # ------------------------------------------------------------------
    # Syntax Errors & Error Handling
    # ------------------------------------------------------------------

    def test_syntax_error_raises_parse_error(self, parser):
        file_path = FIXTURES_DIR / "syntax_error.py"
        with pytest.raises(ParseError) as exc_info:
            parser.parse_file(file_path)

        err = exc_info.value
        assert err.file_path.endswith("syntax_error.py")
        assert err.lineno is not None
        assert isinstance(err.cause, SyntaxError)

    def test_file_size_limit(self):
        tiny_parser = PythonASTParser(max_file_size_bytes=20)
        file_path = FIXTURES_DIR / "simple_function.py"

        with pytest.raises(ParseError) as exc_info:
            tiny_parser.parse_file(file_path)

        err = exc_info.value
        assert "exceeds limit" in err.message

    def test_nonexistent_file(self, parser):
        with pytest.raises(ParseError):
            parser.parse_file("does_not_exist_12345.py")

    def test_relative_path_normalization(self, parser):
        file_path = FIXTURES_DIR / "nested_dir" / "submodule.py"
        pairs = parser.parse_file(file_path, relative_to=FIXTURES_DIR)

        assert len(pairs) == 1
        assert pairs[0].file_path == "nested_dir/submodule.py"

    def test_relative_path_normalization_single_file(self, parser):
        file_path = FIXTURES_DIR / "nested_dir" / "submodule.py"
        # When relative_to is the file path itself, it should normalize to the file name, not '.'
        pairs = parser.parse_file(file_path, relative_to=file_path)

        assert len(pairs) == 1
        assert pairs[0].file_path == "submodule.py"

