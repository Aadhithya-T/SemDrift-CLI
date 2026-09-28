"""
Model manager for resolving, caching, and downloading default SemDrift checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional, Union

from semdrift.model.exceptions import ModelLoadError


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "semdrift" / "models"
DEFAULT_REPO_ID = "Thunderb2314/semdrift-joint-encoder"
DEFAULT_FILENAME = "joint_encoder_checkpoint.pt"


class ModelManager:
    """Manages SemDrift model checkpoint resolution, caching, and downloading.

    Following the resolution flow:
        --checkpoint provided?
             │
      ┌──────┴──────┐
      │             │
     YES            NO
      │             │
    use supplied  ModelManager
     checkpoint         │
                        ▼
                  cached model?
                   /         \
                 yes         no
                  │           │
                load       download
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        model_url: Optional[str] = None,
    ) -> None:
        raw_cache_dir = cache_dir or os.environ.get("SEM_DRIFT_CACHE_DIR")
        self.cache_dir = Path(raw_cache_dir) if raw_cache_dir else DEFAULT_CACHE_DIR
        self.repo_id = repo_id or os.environ.get("SEM_DRIFT_MODEL_REPO", DEFAULT_REPO_ID)
        self.filename = filename or os.environ.get("SEM_DRIFT_MODEL_FILENAME", DEFAULT_FILENAME)
        self.model_url = model_url or os.environ.get("SEM_DRIFT_MODEL_URL")

    @property
    def cached_checkpoint_path(self) -> Path:
        """Return the expected local path of the cached default checkpoint."""
        return self.cache_dir / self.filename

    def is_cached(self) -> bool:
        """Check whether the default model checkpoint exists in the local cache."""
        return self.cached_checkpoint_path.is_file()

    def download_default_checkpoint(self, progress: bool = True) -> Path:
        """Download the default SemDrift model checkpoint to local cache.

        Args:
            progress: Whether to emit download status to stderr.

        Returns:
            Path to the downloaded checkpoint file.

        Raises:
            ModelLoadError: If downloading fails due to network or repository errors.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest_path = self.cached_checkpoint_path

        # If a direct download URL is specified
        if self.model_url:
            if progress:
                sys.stderr.write(f"Downloading default SemDrift model from {self.model_url}...\n")
            try:
                import urllib.request
                urllib.request.urlretrieve(self.model_url, dest_path)
                return dest_path
            except Exception as exc:
                raise ModelLoadError(
                    message=f"Failed to download default model checkpoint from '{self.model_url}': {exc}",
                    checkpoint_path=str(dest_path),
                    cause=exc,
                ) from exc

        # Default: download via huggingface_hub
        if progress:
            sys.stderr.write(
                f"Downloading default SemDrift model '{self.filename}' from Hugging Face ({self.repo_id})...\n"
            )

        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=str(self.cache_dir),
            )
            return Path(downloaded)
        except Exception as exc:
            raise ModelLoadError(
                message=(
                    f"Failed to download default model checkpoint '{self.filename}' from "
                    f"Hugging Face repository '{self.repo_id}': {exc}. "
                    f"You can specify a local checkpoint with --checkpoint <path>."
                ),
                checkpoint_path=str(dest_path),
                cause=exc,
            ) from exc

    def get_default_checkpoint(self, download_if_missing: bool = True) -> Path:
        """Get the default model checkpoint path, downloading if not cached.

        Args:
            download_if_missing: If True, downloads the model if it is not cached.

        Returns:
            Path to the cached or downloaded checkpoint.

        Raises:
            ModelLoadError: If checkpoint is missing and download_if_missing is False,
                           or if download fails.
        """
        if self.is_cached():
            return self.cached_checkpoint_path

        if not download_if_missing:
            raise ModelLoadError(
                message=f"Default model checkpoint not found in cache: {self.cached_checkpoint_path}",
                checkpoint_path=str(self.cached_checkpoint_path),
            )

        return self.download_default_checkpoint()

    def resolve_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Resolve the active checkpoint path following the decision flow:

        If checkpoint_path is provided:
            Returns Path(checkpoint_path).
        If checkpoint_path is None:
            Checks if model is cached:
                If cached -> returns cached checkpoint path.
                If not cached -> downloads model to cache and returns path.

        Args:
            checkpoint_path: Optional user-supplied checkpoint path.

        Returns:
            Resolved Path to the checkpoint file to load.
        """
        if checkpoint_path is not None:
            return Path(checkpoint_path)

        return self.get_default_checkpoint(download_if_missing=True)
