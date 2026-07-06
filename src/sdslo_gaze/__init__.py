"""sdslo-gaze: high-rate 2-D gaze tracking and microsaccade-waveform recovery from SD-SLO.

Public API is intentionally small. The scientific centerpiece is the scanner-flyback contract
in :mod:`sdslo_gaze.scan_timing`; see ``docs/flyback.md`` and ``docs/DESIGN.md``.
"""

from __future__ import annotations

from . import data_io, dynamics, metrics, microsaccade, units
from .gap_model import GapTrack, apply_scan_timing
from .scan_timing import MeasurementState, ScanTiming, TIMING_MODELS, observed_mask
from .strip_tracker import StripTrack, select_strip_width, track

__version__ = "0.1.0"

__all__ = [
    # subpackages
    "units",
    "data_io",
    "dynamics",
    "metrics",
    "microsaccade",
    # flyback contract
    "MeasurementState",
    "ScanTiming",
    "TIMING_MODELS",
    "observed_mask",
    # tracking
    "StripTrack",
    "track",
    "select_strip_width",
    # gap model
    "GapTrack",
    "apply_scan_timing",
    "__version__",
]


def run(*args, **kwargs):
    """Convenience wrapper for :func:`sdslo_gaze.pipeline.run` (imported lazily)."""
    from .pipeline import run as _run

    return _run(*args, **kwargs)
