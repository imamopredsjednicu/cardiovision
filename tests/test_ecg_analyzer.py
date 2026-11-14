"""Testovi za backend/ecg_analyzer.py"""

import numbers

import numpy as np

from backend.ecg_analyzer import calculate_bpm, detect_r_peaks

FS = 360


def make_spike_signal(bpm=60, seconds=10, fs=FS):
    """Stvara umjetni signal s oštrim vrhovima u pravilnim razmacima."""
    signal = np.zeros(seconds * fs)
    step = int(fs * 60 / bpm)
    signal[step // 2::step] = 1.0
    return signal


def test_detect_r_peaks_returns_array_of_indices():
    """detect_r_peaks vraća NumPy polje cijelih brojeva (indeksa)."""
    peaks = detect_r_peaks(make_spike_signal(), fs=FS)

    assert isinstance(peaks, (list, np.ndarray))
    assert np.issubdtype(np.asarray(peaks).dtype, np.integer)
    assert len(peaks) == 10


def test_detect_r_peaks_empty_signal():
    """Prazan signal daje prazno polje vrhova."""
    peaks = detect_r_peaks([], fs=FS)

    assert isinstance(peaks, np.ndarray)
    assert peaks.size == 0


def test_calculate_bpm_returns_number():
    """calculate_bpm vraća numeričku vrijednost."""
    peaks = detect_r_peaks(make_spike_signal(bpm=60), fs=FS)

    bpm = calculate_bpm(peaks, fs=FS)

    assert isinstance(bpm, numbers.Real)
    assert bpm == 60.0


def test_calculate_bpm_from_list():
    """calculate_bpm prihvaća i listu indeksa."""
    bpm = calculate_bpm([0, 180, 360, 540], fs=FS)

    assert isinstance(bpm, numbers.Real)
    assert bpm == 120.0


def test_calculate_bpm_too_few_peaks_returns_none():
    """S manje od dva vrha BPM se ne može izračunati."""
    assert calculate_bpm([], fs=FS) is None
    assert calculate_bpm([100], fs=FS) is None