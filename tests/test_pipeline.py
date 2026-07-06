"""End-to-end integration tests.

The reference-data tests below skip cleanly when the source data tree (SDSLO_DATA_ROOT / the
sibling gaze-model checkout) is not present. ``test_pipeline_synthetic_end_to_end`` needs no data
tree (it builds its own tiny synthetic frame stack) and always runs, so the honesty-critical
observed-only accounting is exercised unconditionally, on any machine.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from sdslo_gaze import data_io, gap_model, metrics, microsaccade, pipeline, strip_tracker
from sdslo_gaze.scan_timing import MeasurementState, ScanTiming, observed_mask


def _data_available() -> bool:
    try:
        data_io.load_reference_strip_track("test1", S=8)
        return True
    except (FileNotFoundError, KeyError):
        return False


_needs_data = pytest.mark.skipif(
    not _data_available(), reason="source data tree not available (set SDSLO_DATA_ROOT)"
)


@_needs_data
def test_pipeline_reference_passes_headline_gates():
    res = pipeline.run("test1", timing_model="flyback10ms", source="reference",
                       out_dir=None, with_dot=True)
    # rate is in the 600-960+ Hz band
    assert res.strip_hz >= 600.0
    # flyback is represented, never as observed motion
    assert "predicted_flyback" in res.role_counts or "missing_flyback" in res.role_counts
    # a physiological microsaccade main sequence is recovered
    assert res.microsaccades["n_events"] >= 10
    slope = res.microsaccades["main_sequence"]["slope"]
    assert 0.4 <= slope <= 1.2
    # held-out correlation vs the pursuit dot is strong
    if res.dot_correlation is not None:
        assert res.dot_correlation["r_x"] >= 0.8


@_needs_data
def test_pipeline_never_counts_nonobserved_as_measured():
    res = pipeline.run("test1", timing_model="flyback10ms", source="reference", out_dir=None,
                       with_dot=False)
    gap = res._gap
    obs = gap.observed()
    # observed() must equal observed_mask(state) and exclude every non-observed state
    assert np.array_equal(obs, observed_mask(gap.state))
    nonobs_states = {
        int(MeasurementState.PREDICTED_FLYBACK),
        int(MeasurementState.MISSING_FLYBACK),
        int(MeasurementState.REJECTED_MISLOCK),
        int(MeasurementState.UNLOCKED),
    }
    assert not any(int(s) in nonobs_states for s in gap.state[obs])
    # n_observed matches the mask
    assert res.n_observed == int(obs.sum())


@_needs_data
def test_timing_sensitivity_changes_composition():
    r10 = pipeline.run("test1", timing_model="flyback10ms", source="reference", out_dir=None,
                       with_dot=False)
    r40 = pipeline.run("test1", timing_model="active40ms", source="reference", out_dir=None,
                       with_dot=False)
    # the (unmeasured) timing assumption is consequential: active-line rate and duty differ
    assert r10.timing["active_line_hz"] != r40.timing["active_line_hz"]
    assert r10.timing["duty_cycle"] > r40.timing["duty_cycle"]  # 10ms flyback -> higher duty


def _make_synthetic_frames(
    n: int = 12, H: int = 200, W: int = 256, fps: float = 14.633, seed: int = 7,
) -> np.ndarray:
    """A fixed random texture, cropped from a bigger canvas at a known smooth per-frame shift.

    Cropping a translated window out of a larger textured canvas is equivalent to the eye
    translating a fixed sensor over the retina: each frame is the same texture, displaced by a
    known, smoothly-varying (dx, dy) in pixels.
    """
    rng = np.random.default_rng(seed)
    margin = 20
    canvas = gaussian_filter(
        rng.standard_normal((H + 2 * margin, W + 2 * margin)).astype(np.float32), sigma=2.0
    )
    f_idx = np.arange(n)
    dx = 2.0 * np.sin(2 * np.pi * f_idx / 6.0)
    dy = 1.5 * np.cos(2 * np.pi * f_idx / 6.0)
    frames = np.empty((n, H, W), dtype=np.float32)
    for i in range(n):
        # Shifting the canvas by -(dx, dy) and re-cropping the same central window is the same
        # as cropping a window translated by +(dx, dy).
        shifted = ndi_shift(canvas, shift=(-dy[i], -dx[i]), order=3, mode="nearest")
        frames[i] = shifted[margin:margin + H, margin:margin + W]
    return frames


def test_pipeline_synthetic_end_to_end():
    """Full stage-by-stage pipeline on synthetic (non-data-tree) frames.

    ``pipeline.run`` loads captures by name, so this drives the same stages directly:
    strip_tracker.track -> gap_model.apply_scan_timing -> microsaccade.detect ->
    metrics.precision_floor. Needs no source data tree, so it runs unconditionally (unlike the
    reference-data tests above) and exercises the observed-only accounting on every machine.
    """
    H, W, S, fps = 200, 256, 8, 14.633
    array = _make_synthetic_frames(n=12, H=H, W=W, fps=fps)
    frames = data_io.Frames(array=array, fps=fps, which="synthetic")

    st = strip_tracker.track(frames, S=S)
    assert st.strip_hz == (W // S) * fps
    assert st.n > 0

    timing = ScanTiming.from_model("flyback10ms", fps=fps, sweeps_per_frame=W, reacq_cols=25)
    gap = gap_model.apply_scan_timing(
        st.t, st.x_arcmin, st.y_arcmin, st.q, st.frame, st.strip, st.S, timing, infov=st.infov,
    )
    obs = gap.observed()
    assert int(obs.sum()) > 0
    assert np.array_equal(obs, observed_mask(gap.state))
    nonobs_states = {
        int(MeasurementState.PREDICTED_FLYBACK),
        int(MeasurementState.MISSING_FLYBACK),
        int(MeasurementState.REJECTED_MISLOCK),
        int(MeasurementState.UNLOCKED),
    }
    assert not any(int(s) in nonobs_states for s in gap.state[obs])

    events = microsaccade.detect(gap.t, gap.x_arcmin, gap.y_arcmin, observed=obs)
    assert all(e.fully_observed for e in events)

    prec = metrics.precision_floor(gap.t[obs], gap.x_arcmin[obs], gap.y_arcmin[obs])
    assert np.isfinite(prec["rms_x"]) and prec["rms_x"] >= 0.0
    assert np.isfinite(prec["rms_y"]) and prec["rms_y"] >= 0.0
