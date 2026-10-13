import numpy as np
import pytest
from selective_hearing.models import calibrate_sources


def test_scale_ambiguous_predictions_are_calibrated_without_clean_input():
    rng = np.random.default_rng(47)
    a, b = rng.normal(0, .08, (2, 16000))
    estimate = np.array([a * 60000, b * -40000])
    corrected = calibrate_sources(estimate, a + b)
    np.testing.assert_allclose(corrected[0], a, atol=1e-7)
    np.testing.assert_allclose(corrected[1], b, atol=1e-7)
    assert np.max(np.abs(corrected)) < .95


def test_zero_predictions_stay_silent_and_invalid_predictions_fail():
    assert np.all(calibrate_sources(np.zeros((2, 30)), np.zeros(30)) == 0)
    with pytest.raises(ValueError):
        calibrate_sources(np.full((2, 30), np.nan), np.zeros(30))
