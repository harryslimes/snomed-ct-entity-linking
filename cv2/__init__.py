"""
Minimal stub for the `cv2` module.

This repo does not require OpenCV for text-only workflows, but some optional
dependencies (e.g. vLLM multimodal modules) import `cv2` at module import time.
In minimal containers, importing the real OpenCV wheel can fail due to missing
system libraries (e.g. `libxcb.so.1`).

This stub allows those imports to succeed. Any attempt to actually use OpenCV
APIs will raise a RuntimeError with a clear message.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.0.0-stub"

# Common constants referenced in optional code paths.
BORDER_CONSTANT = 0
INTER_LINEAR = 1


def __getattr__(name: str) -> Any:
    raise RuntimeError(
        f"`cv2.{name}` was accessed, but OpenCV is not available in this environment. "
        "Install system dependencies for OpenCV (or a working headless build) to use this functionality."
    )


def resize(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
    raise RuntimeError(
        "`cv2.resize` was called, but OpenCV is not available in this environment. "
        "Install system dependencies for OpenCV (or a working headless build) to use this functionality."
    )

