"""Detekcija R-vrhova i izračun srčanog ritma (BPM-a) pomoću SciPy knjižnice."""

import numpy as np
from scipy.signal import find_peaks

# minimalni razmak između dva otkucaja (200 ms ~ 300 BPM)
MIN_RR_SECONDS = 0.2

# udio robusnog maksimuma signala koji vrh mora doseći da bi bio R-vrh
HEIGHT_RATIO = 0.5


def detect_r_peaks(signal, fs=360):
    """Pronalazi R-vrhove u filtriranom EKG signalu.
    Prag visine računa se iz 99. percentila signala umjesto iz maksimuma,
    pa pojedinačni artefakti ne mogu podignuti prag previsoko.
    Vraća NumPy polje indeksa R-vrhova.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.array([], dtype=int)

    height = HEIGHT_RATIO * np.percentile(signal, 99)
    peaks, _ = find_peaks(signal, distance=int(fs * MIN_RR_SECONDS), height=height)
    return peaks


def calculate_bpm(r_peaks, fs=360):
    """Računa prosječni srčani ritam iz razmaka između uzastopnih R-vrhova.
    Vraća BPM zaokružen na jednu decimalu ili None ako je vrhova manje od dva.
    """
    r_peaks = np.asarray(r_peaks)
    if r_peaks.size < 2:
        return None

    rr_intervals = np.diff(r_peaks) / fs
    return round(60.0 / rr_intervals.mean(), 1)


def analyze_ecg_record(filtered_signal, fs=360):
    """Detektira R-vrhove i računa BPM. Vraća rječnik s ključevima r_peaks i bpm."""
    r_peaks = detect_r_peaks(filtered_signal, fs)
    return {
        "r_peaks": r_peaks,
        "bpm": calculate_bpm(r_peaks, fs),
    }


if __name__ == "__main__":
    try:
        from backend.ecg_loader import BASE_DIR, get_ecg_summary
        from backend.signal_processor import process_record_signals
    except ImportError:
        from ecg_loader import BASE_DIR, get_ecg_summary
        from signal_processor import process_record_signals

    summary = get_ecg_summary(BASE_DIR / "data" / "100")
    signals = process_record_signals(summary)
    fs = summary["sample_rate"]
    result = analyze_ecg_record(signals["filtered"], fs)

    print("Zapis:", summary["record_name"])
    print("Broj R-vrhova:", len(result["r_peaks"]))
    print("Prvih 10 R-vrhova:", result["r_peaks"][:10])
    print("BPM:", result["bpm"])