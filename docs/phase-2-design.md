# SemDrift — Phase 2 Design: Repository Scanner & AST Extraction

This document specifies the technical design, data contracts, and architectural decisions for the repository scanning and Python AST extraction layer implemented in Phase 2 of `SemDrift-CLI`.

---

## 1. Core Data Contract: `CodeDocumentPair`

`CodeDocumentPair` is an immutable data structure (`@dataclass(frozen=True)`) that represents a single extracted Python function or method and its associated docstring. It serves as the formal interface between the source extraction layer and downstream model inference / drift detection stages.

### Field Specification

| Field | Type | Description |
|---|---|---|
| `file_path` | `str` | Normalized file path (POSIX-style, relative to scanned repository root where applicable). |
| `qualified_name` | `str` | Fully-qualified deterministic name representing lexical nesting (e.g. `foo`, `Service.run`, `outer.inner`). |
| `line_number` | `int` | 1-indexed line number where the `def` or `async def` statement begins. |
| `end_line_number` | `Optional[int]` | 1-indexed line number where the function definition ends (inclusive). |
| `code` | `str` | Full source code of the function, including decorators and body, preserving original relative indentation. |
| `docstring` | `Optional[str]` | Cleaned docstring text (without enclosing quotes/markers), or `None` if undocumented. |
| `is_method` | `bool` | `True` if the function is defined directly within a class body; `False` otherwise. |
| `is_async` | `bool` | `True` if defined via `async def`; `False` otherwise. |
| `class_name` | `Optional[str]` | Name of the immediate enclosing class if `is_method` is `True`, else `None`. |

### Immutability & Contract Invariants

1. **Frozen Instance**: Fields cannot be reassigned after construction (`FrozenInstanceError` on mutation).
2. **Missing Documentation Representation**: Missing docstrings are represented strictly as `None`. Downstream detectors rely on `docstring is None` to distinguish *undocumented code* from *code with documented but drifted semantics*.
3. **No Synthesized Content**: Neither `code`, `docstring`, nor line numbers are fabricated or defaulted to placeholder strings.

---

## 2. AST Parser: `PythonASTParser`

Located in `src/semdrift/parser/ast_parser.py`.

### API Specification

```python
class PythonASTParser:
    def __init__(self, max_file_size_bytes: int = 1_000_000) -> None: ...

    def parse_source(
        self,
        source: str,
        file_path: str = "<string>",
    ) -> List[CodeDocumentPair]: ...

    def parse_file(
        self,
        file_path: Union[str, Path],
        relative_to: Optional[Union[str, Path]] = None,
    ) -> List[CodeDocumentPair]: ...
```

### AST Traversal & Qualified Name Convention

The parser walks the standard library AST (`ast.parse`) using a recursive scope stack `scope: list[str]`:

- **Top-level functions**: `scope = []` → `qualified_name = func_name`
- **Class methods**: `scope = [ClassName]` → `qualified_name = f"{ClassName}.{method_name}"`, `is_method = True`, `class_name = ClassName`
- **Nested functions**: `scope = [outer_name]` → `qualified_name = f"{outer_name}.{inner_name}"`, `is_method = False`, `class_name = None`
- **Nested methods/helpers**: `scope = [ClassName, method_name]` → `qualified_name = f"{ClassName}.{method_name}.{helper_name}"`
- **Nested classes**: `scope = [OuterClass, InnerClass]` → `qualified_name = f"{OuterClass}.{InnerClass}.{method_name}"`

### Source Code Extraction & Decorator Preservation

Standard Python `ast.get_source_segment()` starts at `node.lineno`, which skips preceding decorators. In `PythonASTParser`:
- If `node.decorator_list` is non-empty, the start line is `node.decorator_list[0].lineno`.
- Otherwise, the start line is `node.lineno`.
- End line is `node.end_lineno`.
- Lines are sliced from the original source file: `source_lines[start_line - 1 : end_line]`.
- This preserves:
  1. All decorator annotations and arguments.
  2. Original relative indentation (e.g. 4-space method indentation inside classes).
  3. The entire executable body without accidental inclusion of sibling AST nodes.

### Docstring Extraction

Docstrings are extracted using standard library `ast.get_docstring(node)`:
- Strips surrounding triple quotes (`"""`, `'''`) and single quotes.
- Cleans leading/trailing indentation in accordance with PEP 257.
- Returns `None` if no docstring statement exists.

### Error Handling: `ParseError`

Syntax errors and unreadable files raise `ParseError(Exception)`:
```python
class ParseError(Exception):
    file_path: str
    message: str
    lineno: Optional[int] = None
    cause: Optional[BaseException] = None
```
`ParseError` carries structured debugging context rather than crashing silently or using bare exceptions.

---

## 3. Repository Scanner: `RepositoryScanner`

Located in `src/semdrift/scanner/repository.py`.

### API Specification

```python
class RepositoryScanner:
    DEFAULT_EXCLUDES: Set[str] = {
        ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "node_modules"
    }

    def __init__(
        self,
        exclude_dirs: Optional[Set[str]] = None,
        max_file_size_bytes: int = 1_000_000,
    ) -> None: ...

    def discover_files(self, target: Union[str, Path]) -> List[Path]: ...
```

### Directory Traversal & Exclusions

- Recursively traverses directories using `os.walk`, modifying `dirs[:]` in-place to ensure excluded directories are never descended into.
- Targets can be a directory (recursively scanned) or a single file (evaluated directly).
- Filters exclusively for Python source files (`*.py`).
- Enforces `max_file_size_bytes` guard (default 1 MB) to prevent denial of service or memory exhaustion from generated/bundled files.
- Returns deterministically sorted `List[Path]`.
- Raises `ScanError` if the target path does not exist or directory access fails.

---

## 4. Research parser → Implementation parser

| Aspect | Research Repository (`SemDrift`) | Implementation Repository (`SemDrift-CLI`) | Rationale |
|---|---|---|---|
| **Component Boundaries** | Combined directory walking and AST parsing in monolithic `ASTParser`. | Strictly decoupled `RepositoryScanner` and `PythonASTParser`. | Separation of concerns; allows scanning independent of parsing and enables alternative scanner backends. |
| **Data Contract** | `FunctionInfo` dataclass (mutable, included `params`, `raises`, `return_annotation`). | `CodeDocumentPair` (`@dataclass(frozen=True)`). | Clean contract with immutability guarantees tailored for downstream embedding and drift detection. |
| **Missing Docstring** | `docstring: str = ""` and `has_docstring: bool`. | `docstring: Optional[str] = None`. | Clear semantic distinction: `None` represents absent documentation; non-empty strings represent documentation to evaluate for drift. |
| **Docstring Removal in Code** | Stripped docstrings from function source code using line slicing. | Preserved full source code including docstring and decorators. | Code representation retains complete structural context; downstream tokenizers handle docstring/code separation as needed without corrupting source integrity. |
| **Decorator Extraction** | Extracted decorator names as strings via `ast.unparse(dec)`. Source code started after decorators. | Source code slice starts at `decorator_list[0].lineno`, preserving full decorators and indentation. | Ensures downstream embedding models observe decorator semantics (e.g. `@property`, `@override`, `@deprecated`, custom validation decorators). |
| **Qualified Names** | Flat `class_name` attribute only (`Calculator.add`). No support for nested functions. | Lexical scope stack: hierarchical deterministic names (`outer.inner`, `Class.method.helper`). | Comprehensive coverage of modern Python code patterns where closures and nested functions are common. |
| **Tree-sitter / Multi-language** | Superseded AST parser with tree-sitter `UniversalParser` in V2. | Pure Python standard library `ast`. | Zero compiled C extensions or heavy binary dependencies required for Python semantic drift detection. |
| **Error Handling** | Caught `SyntaxError` and `OSError` returning empty list `[]`. | Explicit `ParseError` and `ScanError` carrying structured failure context. | Enables CLI and reporter to record skipped files, line numbers, and error reasons without terminating entire scans. |
| **Dependencies** | Mixed runtime dependencies with `torch`, `transformers`, `tree-sitter`. | Zero external runtime dependencies (Python standard library only). | Fast installation, minimal attack surface, clean developer tool architecture. |

---

## 5. Known Limitations

1. **Python Only**: Currently only analyzes Python (`*.py`) source files. Jupyter notebooks (`*.ipynb`), Cython, and other languages are outside Phase 2 scope.
2. **Standard Library AST**: Highly dynamic source code or syntax valid only in experimental Python versions may raise `ParseError`.
3. **Encoding**: Expects UTF-8 source files per PEP 3120 standard. Non-UTF-8 files with non-standard legacy encodings raise a structured `ParseError`.
