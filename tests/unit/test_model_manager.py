"""
Unit tests for ModelManager resolution, caching, and downloading flows.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from semdrift.model.exceptions import ModelLoadError
from semdrift.model.manager import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FILENAME,
    DEFAULT_REPO_ID,
    ModelManager,
)


class TestModelManager:
    """Verify ModelManager cache detection, resolution logic, and error handling."""

    def test_default_initialization(self):
        manager = ModelManager()
        assert manager.cache_dir == DEFAULT_CACHE_DIR
        assert manager.repo_id == DEFAULT_REPO_ID
        assert manager.filename == DEFAULT_FILENAME
        assert manager.model_url is None
        assert manager.cached_checkpoint_path == DEFAULT_CACHE_DIR / DEFAULT_FILENAME

    def test_custom_initialization(self, tmp_path: Path):
        manager = ModelManager(
            cache_dir=tmp_path / "custom_cache",
            repo_id="custom/repo",
            filename="custom_model.pt",
            model_url="https://example.com/model.pt",
        )
        assert manager.cache_dir == tmp_path / "custom_cache"
        assert manager.repo_id == "custom/repo"
        assert manager.filename == "custom_model.pt"
        assert manager.model_url == "https://example.com/model.pt"
        assert manager.cached_checkpoint_path == tmp_path / "custom_cache" / "custom_model.pt"

    def test_env_var_overrides(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("SEM_DRIFT_CACHE_DIR", str(tmp_path / "env_cache"))
        monkeypatch.setenv("SEM_DRIFT_MODEL_REPO", "env/repo")
        monkeypatch.setenv("SEM_DRIFT_MODEL_FILENAME", "env_model.pt")
        monkeypatch.setenv("SEM_DRIFT_MODEL_URL", "https://env.example.com/model.pt")

        manager = ModelManager()
        assert manager.cache_dir == tmp_path / "env_cache"
        assert manager.repo_id == "env/repo"
        assert manager.filename == "env_model.pt"
        assert manager.model_url == "https://env.example.com/model.pt"

    def test_is_cached_returns_false_when_absent(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path)
        assert manager.is_cached() is False

    def test_is_cached_returns_true_when_present(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path, filename="model.pt")
        ckpt_file = tmp_path / "model.pt"
        ckpt_file.write_text("model weights", encoding="utf-8")
        assert manager.is_cached() is True

    def test_resolve_checkpoint_with_supplied_path_returns_path(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path)
        custom_path = tmp_path / "my_custom_model.pt"
        resolved = manager.resolve_checkpoint(custom_path)
        assert resolved == custom_path

    def test_resolve_checkpoint_when_none_and_cached_returns_cached_path(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path, filename="default.pt")
        cached_file = tmp_path / "default.pt"
        cached_file.write_text("cached weights", encoding="utf-8")

        resolved = manager.resolve_checkpoint(None)
        assert resolved == cached_file

    def test_resolve_checkpoint_when_none_and_not_cached_downloads(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path, filename="default.pt")
        expected_dest = tmp_path / "default.pt"

        def fake_download(*args, **kwargs):
            expected_dest.write_text("downloaded weights", encoding="utf-8")
            return expected_dest

        with patch.object(manager, "download_default_checkpoint", side_effect=fake_download) as mock_dl:
            resolved = manager.resolve_checkpoint(None)
            assert resolved == expected_dest
            mock_dl.assert_called_once()

    def test_get_default_checkpoint_download_if_missing_false_raises_when_missing(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path, filename="missing.pt")
        with pytest.raises(ModelLoadError) as exc_info:
            manager.get_default_checkpoint(download_if_missing=False)
        assert "Default model checkpoint not found in cache" in str(exc_info.value)

    def test_download_via_hf_hub_success(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path, repo_id="org/repo", filename="model.pt")
        dest_file = tmp_path / "model.pt"

        with patch("huggingface_hub.hf_hub_download", return_value=str(dest_file)) as mock_hf:
            result = manager.download_default_checkpoint(progress=False)
            assert result == dest_file
            mock_hf.assert_called_once_with(
                repo_id="org/repo",
                filename="model.pt",
                local_dir=str(tmp_path),
            )

    def test_download_via_hf_hub_failure_raises_model_load_error(self, tmp_path: Path):
        manager = ModelManager(cache_dir=tmp_path, repo_id="org/repo", filename="model.pt")

        with patch("huggingface_hub.hf_hub_download", side_effect=RuntimeError("Connection refused")):
            with pytest.raises(ModelLoadError) as exc_info:
                manager.download_default_checkpoint(progress=False)
            assert "Failed to download default model checkpoint" in str(exc_info.value)
            assert "Connection refused" in str(exc_info.value)

    def test_download_via_url_success(self, tmp_path: Path):
        manager = ModelManager(
            cache_dir=tmp_path,
            filename="model.pt",
            model_url="https://example.com/weights.pt",
        )
        dest_file = tmp_path / "model.pt"

        def fake_urlretrieve(url, filename):
            Path(filename).write_text("retrieved content", encoding="utf-8")

        with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock_retrieve:
            result = manager.download_default_checkpoint(progress=False)
            assert result == dest_file
            assert dest_file.exists()
            mock_retrieve.assert_called_once_with("https://example.com/weights.pt", dest_file)

    def test_download_via_url_failure_raises_model_load_error(self, tmp_path: Path):
        manager = ModelManager(
            cache_dir=tmp_path,
            filename="model.pt",
            model_url="https://example.com/weights.pt",
        )

        with patch("urllib.request.urlretrieve", side_effect=OSError("Network down")):
            with pytest.raises(ModelLoadError) as exc_info:
                manager.download_default_checkpoint(progress=False)
            assert "Failed to download default model checkpoint from 'https://example.com/weights.pt'" in str(
                exc_info.value
            )
