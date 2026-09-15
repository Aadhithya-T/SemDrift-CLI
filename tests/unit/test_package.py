"""Tests for the semdrift package foundation."""

import importlib


def test_import_semdrift():
    """Verify that `import semdrift` works after installation."""
    import semdrift  # noqa: F811

    assert semdrift is not None


def test_version_exists():
    """Verify that semdrift exposes a __version__ string."""
    import semdrift

    assert hasattr(semdrift, "__version__")
    assert isinstance(semdrift.__version__, str)
    assert len(semdrift.__version__) > 0


def test_submodules_importable():
    """Verify that all planned submodules can be imported."""
    submodules = [
        "semdrift.parser",
        "semdrift.scanner",
        "semdrift.model",
        "semdrift.detection",
        "semdrift.reporting",
        "semdrift.cli",
    ]
    for module_name in submodules:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import {module_name}"


def test_public_api_exports():
    """Verify that semdrift re-exports core public classes."""
    import semdrift

    assert hasattr(semdrift, "CodeDocumentPair")
    assert hasattr(semdrift, "PythonASTParser")
    assert hasattr(semdrift, "ParseError")
    assert hasattr(semdrift, "RepositoryScanner")
    assert hasattr(semdrift, "ScanError")
    assert hasattr(semdrift, "ModelConfig")
    assert hasattr(semdrift, "ModelPrediction")
    assert hasattr(semdrift, "SemDriftModel")
    assert hasattr(semdrift, "ModelError")
    assert hasattr(semdrift, "ModelLoadError")
    assert hasattr(semdrift, "ModelInferenceError")
    assert hasattr(semdrift, "DetectionConfig")
    assert hasattr(semdrift, "DriftResult")
    assert hasattr(semdrift, "DriftDetector")
    assert hasattr(semdrift, "DetectionError")
    assert hasattr(semdrift, "DetectionConfigError")
    assert hasattr(semdrift, "DetectionInputError")

