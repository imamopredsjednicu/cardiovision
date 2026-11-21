"""Testovi za backend/arrhythmia_classifier.py"""

import numpy as np
import pytest

from backend.arrhythmia_classifier import (
    detect_arrhythmias,
    extract_rr_features,
    label_peaks_from_annotations,
)

FS = 360
SIGNAL = np.zeros(10 * FS)


@pytest.mark.parametrize("peaks", [[], [100], None])
def test_extract_rr_features_too_few_peaks(peaks):
    """S manje od dva vrha vraća se prazna matrica oblika (0, 2)."""
    assert extract_rr_features(peaks, FS).shape == (0, 2)


@pytest.mark.parametrize("peaks", [[], [100], None, [np.nan, np.inf]])
def test_detect_arrhythmias_too_few_peaks(peaks):
    """S 0 ili 1 ispravnim vrhom nema aritmija."""
    assert detect_arrhythmias(SIGNAL, peaks, FS) == ([], 0.0)


def test_detect_arrhythmias_without_signal():
    """Bez signala nijedan vrh nije unutar granica."""
    assert detect_arrhythmias(None, [100, 460], FS) == ([], 0.0)
    assert detect_arrhythmias([], [100, 460], FS) == ([], 0.0)


def test_detect_arrhythmias_duplicate_peaks():
    """Duplikati vrhova ne uzrokuju dijeljenje nulom (RR = 0)."""
    peaks = [360, 360, 720, 720, 1080, 1440]

    indices, ratio = detect_arrhythmias(SIGNAL, peaks, FS)

    assert indices == []
    assert ratio == 0.0


def test_detect_arrhythmias_finds_premature_beat():
    """Prerani otkucaj u pravilnom ritmu označen je kao aritmija."""
    peaks = [360, 720, 1080, 1250, 1800, 2160, 2520]

    indices, ratio = detect_arrhythmias(SIGNAL, peaks, FS)

    assert 1250 in indices
    assert 0 < ratio <= 1


@pytest.mark.parametrize("annotations", [
    None,
    {},
    {"sample": [], "symbol": []},
    {"sample": [5, 10], "symbol": ["+", "~"]},
])
def test_label_peaks_without_beat_annotations(annotations):
    """Bez anotacija otkucaja svi vrhovi dobivaju oznaku -1."""
    labels = label_peaks_from_annotations([100, 460], annotations, FS)

    assert labels.tolist() == [-1, -1]


def test_label_peaks_single_annotation():
    """Jedna anotacija pridružuje se samo vrhu u njezinoj blizini."""
    labels = label_peaks_from_annotations([100, 2000], {"sample": [105], "symbol": ["V"]}, FS)

    assert labels.tolist() == [1, -1]


def test_label_peaks_no_peaks():
    """Bez vrhova vraća se prazno polje oznaka."""
    labels = label_peaks_from_annotations([], {"sample": [105], "symbol": ["N"]}, FS)

    assert labels.size == 0
