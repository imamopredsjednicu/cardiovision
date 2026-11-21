"""Testovi za backend/ecg_analyzer.py"""

import numbers

import numpy as np
import pytest

from backend.ecg_analyzer import analyze_ecg_record, calculate_bpm, detect_r_peaks

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

def test_detect_r_peaks_ignores_nan_and_inf():
    """NaN uzorci ne ruše detekciju, a Inf se ne proglašava R-vrhom."""
    signal = make_spike_signal()
    signal[5] = np.inf
    signal[100] = np.nan

    peaks = detect_r_peaks(signal, fs=FS)

    assert len(peaks) == 10
    assert 5 not in peaks


def test_detect_r_peaks_short_or_invalid_signal():
    """Prekratak signal, signal od samih NaN-ova i None daju prazno polje."""
    assert detect_r_peaks([1.0], fs=FS).size == 0
    assert detect_r_peaks(np.full(1000, np.nan), fs=FS).size == 0
    assert detect_r_peaks(None, fs=FS).size == 0


def test_detect_r_peaks_low_fs_does_not_crash():
    """Niska frekvencija (razmak vrhova < 1 uzorak) ne smije srušiti find_peaks."""
    peaks = detect_r_peaks([0, 1, 0, 1, 0, 1, 0], fs=1)

    assert list(peaks) == [1, 3, 5]


def test_calculate_bpm_duplicate_and_invalid_peaks():
    """Duplikati i NaN vrhovi ne uzrokuju dijeljenje nulom."""
    assert calculate_bpm([100, 100, 100], fs=FS) is None
    assert calculate_bpm([0, np.nan, 360, 360], fs=FS) == 60.0
    assert calculate_bpm(None, fs=FS) is None


def test_calculate_bpm_invalid_fs_raises():
    """Neispravna frekvencija uzorkovanja podiže ValueError."""
    with pytest.raises(ValueError):
        calculate_bpm([0, 360], fs=0)


def test_analyze_ecg_record_flat_signal():
    """Ravna linija daje 0 vrhova i BPM None."""
    result = analyze_ecg_record(np.zeros(3600), fs=FS)

    assert result["r_peaks"].size == 0
    assert result["bpm"] is None
