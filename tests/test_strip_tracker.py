"""Tests for :mod:`sdslo_gaze.strip_tracker`."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
import pytest

from sdslo_gaze import data_io, units
from sdslo_gaze.strip_tracker import select_strip_width, track


# --- select_strip_width ------------------------------------------------------

def test_select_strip_width_meets_target_rate():
    W, fps, target = 808, 14.633, 960.0
    S = select_strip_width(W, fps, target)
    realized = (W // S) * fps
    assert realized >= target
    # No wider strip should also clear the target (S should be the widest that clears it).
    assert (W // (S + 1)) * fps < target


def test_select_strip_width_monotonic_with_target():
    W, fps = 808, 14.633
    S_low = select_strip_width(W, fps, target_hz=300.0)
    S_mid = select_strip_width(W, fps, target_hz=960.0)
    S_high = select_strip_width(W, fps, target_hz=3000.0)
    # Higher target rate -> narrower (or equal) strips.
    assert S_low >= S_mid >= S_high


def test_select_strip_width_fallback_when_unreachable():
    # target far beyond what any non-overlapping strip width can reach.
    W, fps = 808, 14.633
    S = select_strip_width(W, fps, target_hz=1e9)
    assert 1 <= S <= W


# --- track() on a synthetic shifted-texture stack ---------------------------

def _make_shifted_stack(dx_off, dy_off, H=200, W=160, margin=20, seed=0):
    """Build a frame stack by cropping a fixed noise canvas at integer offsets.

    Frame f = canvas[margin+dy_off[f] : +H, margin+dx_off[f] : +W]. Because every
    frame is an exact crop of the SAME canvas (no independent per-frame noise), the
    frame-to-frame NCC match is (near-)exact and the recovered displacement should
    reproduce the injected offsets to well under a pixel.
    """
    rng = np.random.default_rng(seed)
    canvas = rng.standard_normal((H + 2 * margin, W + 2 * margin)).astype(np.float32)
    frames = np.stack([
        canvas[margin + dy: margin + dy + H, margin + dx: margin + dx + W]
        for dx, dy in zip(dx_off, dy_off)
    ])
    return frames


def test_track_recovers_injected_shift():
    dx_off = [0, 3, 5, 5, 8, 8]
    dy_off = [0, -2, -2, 1, 1, 3]
    fps = 20.0
    S = 20  # 160 // 20 = 8 strips, no remainder
    frames = _make_shifted_stack(dx_off, dy_off)

    tr = track(frames, fps=fps, S=S)

    W = 160
    assert tr.S == S
    assert tr.W == W
    assert tr.H == 200
    assert tr.fps == fps
    assert tr.strip_hz == pytest.approx((W // S) * fps)

    # A uniform whole-frame shift -> every strip in a frame should carry the same
    # cumulative displacement (no meaningful within-frame residual).
    for f in range(1, len(dx_off)):
        mask = tr.frame == f
        assert mask.any()
        expected_x = dx_off[f] - dx_off[0]
        expected_y = dy_off[f] - dy_off[0]
        assert np.allclose(tr.x_px[mask], expected_x, atol=0.5)
        assert np.allclose(tr.y_px[mask], expected_y, atol=0.5)

    # arcmin conversion matches units.py exactly (axis-specific scale).
    assert np.allclose(tr.x_arcmin, units.col_px_to_arcmin(tr.x_px))
    assert np.allclose(tr.y_arcmin, units.row_px_to_arcmin(tr.y_px))


def test_track_accepts_frames_object():
    dx_off = [0, 2, 4]
    dy_off = [0, 1, 1]
    fps = 15.0
    frames = _make_shifted_stack(dx_off, dy_off)
    wrapped = data_io.Frames(array=frames, fps=fps, which="synthetic")

    tr = track(wrapped, S=20)

    assert tr.fps == fps
    assert tr.n == (len(dx_off) - 1) * (160 // 20)


# --- optional cross-check against the research repo's cached reference ------

def test_track_matches_reference_cache_if_available():
    try:
        ref = data_io.load_reference_strip_track("test1", S=8)
        frames = data_io.load_frames("test1", max_frames=40)
    except FileNotFoundError:
        pytest.skip("reference strip-track cache / test1 frames not available")
        return

    tr = track(frames, S=8, pad=64)

    ref_frame = np.asarray(ref["frame"])
    keep = ref_frame < frames.n
    ref_along = np.asarray(ref["along_px"])[keep]
    ref_infov = np.asarray(ref["infov"])[keep].astype(bool)

    n = min(len(ref_along), tr.n)
    both_infov = tr.infov[:n] & ref_infov[:n]
    if both_infov.sum() < 10:
        pytest.skip("insufficient in-FOV overlap between reference and reproduction")
        return

    r = np.corrcoef(tr.x_px[:n][both_infov], ref_along[:n][both_infov])[0, 1]
    assert r > 0.9
