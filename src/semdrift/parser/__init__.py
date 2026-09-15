"""
semdrift.parser — Python source code parsing and AST extraction.

Provides AST-based extraction of functions and methods into immutable
CodeDocumentPair records without external ML or heavy parser dependencies.
"""

from semdrift.parser.models import CodeDocumentPair, ParseError
from semdrift.parser.ast_parser import PythonASTParser

__all__ = [
    "CodeDocumentPair",
    "ParseError",
    "PythonASTParser",
]
