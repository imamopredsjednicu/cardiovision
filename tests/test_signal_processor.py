"""Testovi za backend/signal_processor.py"""

import numpy as np
import pytest

from backend.ecg_loader import BASE_DIR, get_ecg_summary
from backend.signal_processor import apply_bandpass_filter, process_record_signals

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

def test_apply_bandpass_filter_empty_signal():
    """Prazan signal vraća prazno polje umjesto rušenja."""
    assert apply_bandpass_filter([]).size == 0
    assert apply_bandpass_filter(None).size == 0


def test_apply_bandpass_filter_short_signal_returned_unfiltered():
    """Signal prekratak za filtfilt vraća se nefiltriran, istog oblika."""
    data = [0.1, 0.5, 1.0, 0.5, 0.1]

    filtered = apply_bandpass_filter(data)

    assert np.allclose(filtered, data)


def test_apply_bandpass_filter_handles_nan_and_inf():
    """NaN/Inf uzorci se interpoliraju, pa je cijeli izlaz konačan."""
    signal = np.sin(np.linspace(0, 20 * np.pi, 2000))
    signal[[10, 500, 501, 1999]] = [np.nan, np.inf, -np.inf, np.nan]

    filtered = apply_bandpass_filter(signal)

    assert filtered.shape == signal.shape
    assert np.all(np.isfinite(filtered))


def test_apply_bandpass_filter_all_nan_gives_zeros():
    """Signal bez ijednog ispravnog uzorka daje nule."""
    filtered = apply_bandpass_filter(np.full(1000, np.nan))

    assert np.all(filtered == 0)


@pytest.mark.parametrize("fs", [0, -360, np.nan, None])
def test_apply_bandpass_filter_invalid_fs_raises(fs):
    """Neispravna frekvencija uzorkovanja podiže ValueError, a ne ZeroDivisionError."""
    with pytest.raises(ValueError):
        apply_bandpass_filter(np.zeros(1000), fs=fs)


def test_process_record_signals_missing_signal():
    """Sažetak bez signala daje prazne nizove."""
    result = process_record_signals({"sample_rate": 360})

    assert result["raw"].size == 0
    assert result["filtered"].size == 0
