"""Klasifikacija aritmija iz RR-intervala pomoću scikit-learn knjižnice."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier

# dopušteno relativno odstupanje RR-intervala od lokalnog medijana (20 %)
RR_TOLERANCE = 0.2

# broj okolnih otkucaja iz kojih se računa lokalni medijan RR-intervala
LOCAL_WINDOW = 9

# oznake MIT-BIH anotacija koje se smatraju normalnim otkucajima
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}

# sve oznake otkucaja (ostale anotacije su promjene ritma, šum i sl.)
BEAT_SYMBOLS = NORMAL_SYMBOLS | {"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"}


def extract_rr_features(r_peaks, fs=360):
    """Računa prethodni i sljedeći RR-interval (u sekundama) za svaki R-vrh.
    Prvi vrh nema prethodnog, a zadnji sljedećeg intervala, pa se za njih
    uzima najbliži postojeći interval.
    Vraća NumPy matricu oblika (broj_vrhova, 2) sa stupcima [pre_rr, post_rr].
    """
    r_peaks = np.asarray(r_peaks)
    if r_peaks.size < 2:
        return np.empty((0, 2))

    rr = np.diff(r_peaks) / fs
    pre_rr = np.concatenate(([rr[0]], rr))
    post_rr = np.concatenate((rr, [rr[-1]]))
    return np.column_stack((pre_rr, post_rr))


def train_simple_model(features, labels):
    """Inicijalizira i trenira RandomForestClassifier na RR-značajkama.
    Aritmični otkucaji su rijetki, pa se klase uravnotežuju težinama.
    """
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    model.fit(features, labels)
    return model


def _local_median(values, window=LOCAL_WINDOW):
    """Medijan svake vrijednosti i njezinih susjeda (rubovi se nadopunjuju)."""
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.median(sliding_window_view(padded, window), axis=1)


def detect_arrhythmias(signal, r_peaks, fs=360):
    """Označava otkucaje čiji prethodni RR-interval odstupa od lokalnog
    medijana više od RR_TOLERANCE (prerani ili zakašnjeli otkucaji).
    Vraća listu indeksa uzoraka R-vrhova s aritmijom, udio aritmičnih otkucaja.
    """
    r_peaks = np.asarray(r_peaks)
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(signal))]
    if r_peaks.size < 2:
        return [], 0.0

    pre_rr = extract_rr_features(r_peaks, fs)[:, 0]
    expected = _local_median(pre_rr)
    deviation = np.abs(pre_rr - expected) / expected

    arrhythmic = r_peaks[deviation > RR_TOLERANCE]
    return arrhythmic.tolist(), len(arrhythmic) / r_peaks.size


def label_peaks_from_annotations(r_peaks, annotations, fs=360, max_offset=0.1):
    """Svakom R-vrhu pridružuje najbližu anotaciju unutar max_offset sekundi.
    Vraća polje oznaka: 0 = normalan otkucaj, 1 = aritmija, -1 = bez anotacije.
    """
    r_peaks = np.asarray(r_peaks)
    samples = np.asarray(annotations["sample"])
    symbols = np.asarray(annotations["symbol"])

    is_beat = np.isin(symbols, list(BEAT_SYMBOLS))
    samples, symbols = samples[is_beat], symbols[is_beat]

    pos = np.clip(np.searchsorted(samples, r_peaks), 1, len(samples) - 1)
    left, right = samples[pos - 1], samples[pos]
    nearest = np.where(r_peaks - left <= right - r_peaks, pos - 1, pos)

    labels = np.where(np.isin(symbols[nearest], list(NORMAL_SYMBOLS)), 0, 1)
    labels[np.abs(samples[nearest] - r_peaks) > max_offset * fs] = -1
    return labels


if __name__ == "__main__":
    try:
        from backend.ecg_analyzer import analyze_ecg_record
        from backend.ecg_loader import BASE_DIR, get_ecg_summary, load_ecg_annotations
        from backend.signal_processor import process_record_signals
    except ImportError:
        from ecg_analyzer import analyze_ecg_record
        from ecg_loader import BASE_DIR, get_ecg_summary, load_ecg_annotations
        from signal_processor import process_record_signals

    record_path = BASE_DIR / "data" / "100"
    summary = get_ecg_summary(record_path)
    fs = summary["sample_rate"]
    filtered = process_record_signals(summary)["filtered"]
    r_peaks = analyze_ecg_record(filtered, fs)["r_peaks"]

    indices, ratio = detect_arrhythmias(filtered, r_peaks, fs)
    print("Zapis:", summary["record_name"])
    print("Broj R-vrhova:", len(r_peaks))
    print("Broj pronađenih aritmija:", len(indices))
    print(f"Udio aritmičnih otkucaja: {ratio * 100:.2f} %")

    annotations = load_ecg_annotations(record_path)
    if annotations is not None:
        features = extract_rr_features(r_peaks, fs)
        labels = label_peaks_from_annotations(r_peaks, annotations, fs)
        known = labels >= 0
        model = train_simple_model(features[known], labels[known])
        predicted = model.predict(features[known])

        print("Aritmije prema anotacijama:", int(labels[known].sum()))
        print("Aritmije prema RandomForest modelu:", int(predicted.sum()))