"""Filtriranje EKG signala pomoću SciPy knjižnice."""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):
    """Računa koeficijente (b, a) Butterworthovog pojasnog filtera."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if not 0 < low < high < 1:
        raise ValueError(
            f"Neispravne granice filtera ({lowcut}-{highcut} Hz) za frekvenciju {fs} Hz."
        )
    b, a = butter(order, [low, high], btype="band")
    return b, a


def apply_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=4):
    """Primjenjuje pojasni filter na sirovi EKG signal.
    filtfilt filtrira unaprijed i unatrag pa nema faznog kašnjenja.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, np.asarray(data, dtype=float))


def process_record_signals(ecg_summary_dict):
    """Filtrira kanal 0 iz sažetka ecg_loader modula.
    Vraća rječnik sa sirovim (raw) i filtriranim (filtered) signalom.
    """
    raw = np.asarray(ecg_summary_dict["channel_0_signal"], dtype=float)
    fs = ecg_summary_dict["sample_rate"]
    filtered = apply_bandpass_filter(raw, fs=fs)

    return {
        "raw": raw,
        "filtered": filtered,
    }


if __name__ == "__main__":
    try:
        from backend.ecg_loader import BASE_DIR, get_ecg_summary
    except ImportError:
        from ecg_loader import BASE_DIR, get_ecg_summary

    summary = get_ecg_summary(BASE_DIR / "data" / "100")
    signals = process_record_signals(summary)
    filtered = signals["filtered"]

    print("Zapis:", summary["record_name"])
    print("Duljina filtriranog signala:", len(filtered))
    print(f"Min: {filtered.min():.4f}")
    print(f"Max: {filtered.max():.4f}")
    print(f"Srednja vrijednost: {filtered.mean():.4f}")
    print(f"Standardna devijacija: {filtered.std():.4f}")
