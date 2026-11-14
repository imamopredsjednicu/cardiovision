"""Testovi za backend/signal_processor.py"""

import numpy as np
import pytest

from backend.ecg_loader import BASE_DIR, get_ecg_summary
from backend.signal_processor import apply_bandpass_filter

RECORD_100 = BASE_DIR / "data" / "100"


def test_apply_bandpass_filter_keeps_shape_on_synthetic_signal():
    """Izlaz filtera mora imati isti oblik kao ulazni niz."""
    fs = 360
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 60 * t)

    filtered = apply_bandpass_filter(signal, fs=fs)

    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == signal.shape


def test_apply_bandpass_filter_accepts_list():
    """Filter prihvaća i običnu Python listu."""
    data = list(np.random.default_rng(0).normal(size=1000))

    filtered = apply_bandpass_filter(data)

    assert filtered.shape == (len(data),)


def test_apply_bandpass_filter_invalid_cutoff_raises():
    """Granice izvan Nyquistove frekvencije moraju podići ValueError."""
    with pytest.raises(ValueError):
        apply_bandpass_filter(np.zeros(1000), lowcut=0.5, highcut=200.0, fs=360)


@pytest.mark.skipif(not (RECORD_100.parent / "100.hea").is_file(), reason="nema zapisa data/100")
def test_apply_bandpass_filter_keeps_shape_on_record_100():
    """Filtriranje stvarnog zapisa 100 ne smije promijeniti oblik signala."""
    summary = get_ecg_summary(RECORD_100)
    raw = summary["channel_0_signal"]

    filtered = apply_bandpass_filter(raw, fs=summary["sample_rate"])

    assert filtered.shape == raw.shape
    assert np.all(np.isfinite(filtered))