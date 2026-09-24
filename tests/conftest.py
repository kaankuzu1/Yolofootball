"""Shared fixtures: a real video file on disk, and a stock config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running the suite straight from a checkout, without installing.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_analysis.config import Config  # noqa: E402
from football_analysis.io.synthetic import make_sample_clip  # noqa: E402


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory: pytest.TempPathFactory):
    """A 6-second synthetic 1v1 clip, rendered once for the whole session."""
    path = tmp_path_factory.mktemp("clips") / "sample.mp4"
    truth = make_sample_clip(path, duration_s=6.0, fps=25.0, width=640, height=360)
    return path, truth


@pytest.fixture
def config() -> Config:
    """Defaults, trimmed so tests stay fast."""
    cfg = Config()
    cfg.video.resize_width = 320
    return cfg.validate()
