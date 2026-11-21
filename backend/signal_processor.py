"""Filtriranje EKG signala pomoću SciPy knjižnice."""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs, order=4):
    """Računa koeficijente (b, a) Butterworthova pojasnog filtera."""
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Frekvencija uzorkovanja mora biti pozitivan broj, a ne {fs}.")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if not 0 < low < high < 1:
        raise ValueError(
            f"Neispravne granice filtera ({lowcut}-{highcut} Hz) za frekvenciju {fs} Hz."
        )
    b, a = butter(order, [low, high], btype="band")
    return b, a


def _fill_non_finite(signal):
    """Zamjenjuje NaN/Inf uzorke linearnom interpolacijom između susjednih
    ispravnih uzoraka; ako ispravnih uzoraka nema, vraća niz nula.
    """
    finite = np.isfinite(signal)
    if finite.all():
        return signal
    if not finite.any():
        return np.zeros_like(signal)

    idx = np.arange(signal.size)
    filled = signal.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], signal[finite])
    return filled


def apply_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=4):
    """Primjenjuje pojasni filter na sirovi EKG signal.
    filtfilt filtrira unaprijed i unatrag, pa nema faznog kašnjenja.
    NaN/Inf uzorci se prije filtriranja interpoliraju, prazan signal vraća se
    kao prazno polje, a signal prekratak za filtfilt vraća se nefiltriran.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order)

    signal = np.atleast_1d(np.asarray([] if data is None else data, dtype=float))
    if signal.size == 0:
        return np.array([], dtype=float)

    if signal.ndim == 1:
        signal = _fill_non_finite(signal)
    else:
        signal = np.apply_along_axis(_fill_non_finite, -1, signal)

    # filtfilt zahtijeva signal dulji od duljine nadopune (padlen)
    if signal.shape[-1] <= 3 * max(len(a), len(b)):
        return signal.copy()
    return filtfilt(b, a, signal)


def process_record_signals(ecg_summary_dict):
    """Filtrira kanal 0 iz sažetka ecg_loader modula.
    Vraća rječnik sa sirovim (raw) i filtriranim (filtered) signalom.
    Ako signal nedostaje, oba su niza prazna; bez sample_rate podiže ValueError.
    """
    ecg_summary_dict = ecg_summary_dict or {}
    signal = ecg_summary_dict.get("channel_0_signal")
    raw = np.atleast_1d(np.asarray([] if signal is None else signal, dtype=float))
    fs = ecg_summary_dict.get("sample_rate")
    if fs is None:
        raise ValueError("Sažetak zapisa ne sadrži frekvenciju uzorkovanja (sample_rate).")
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
