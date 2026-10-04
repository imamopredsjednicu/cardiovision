"""Učitavanje i obrada PhysioNet EKG zapisa pomoću wfdb knjižnice."""

from pathlib import Path

import wfdb

# korijenska mapa projekta
BASE_DIR = Path(__file__).resolve().parent.parent

# ekstenzije koje wfdb ne želi u putanji (očekuje samo naziv zapisa)
WFDB_EXTENSIONS = {".dat", ".hea", ".atr"}


def _record_path(file_path):
    """Vraća putanju do zapisa bez ekstenzije.
    Podiže FileNotFoundError ako pripadajuća .hea datoteka ne postoji.
    """
    path = Path(file_path)
    if path.suffix.lower() in WFDB_EXTENSIONS:
        path = path.with_suffix("")

    header = path.parent / f"{path.name}.hea"
    if not header.is_file():
        raise FileNotFoundError(f"EKG zapis '{path}' ne postoji (nedostaje {header}).")
    return path


def load_ecg_record(file_path):
    """Učitava EKG zapis i vraća rječnik sa signalom i osnovnim podatcima."""
    path = _record_path(file_path)
    try:
        record = wfdb.rdrecord(str(path))
    except Exception as e:
        raise ValueError(f"Greška pri čitanju EKG zapisa '{path}': {e}") from e

    if record.p_signal is None:
        raise ValueError(f"EKG zapis '{path}' ne sadrži fizički signal.")

    return {
        "p_signal": record.p_signal,
        "fs": record.fs,
        "sig_name": record.sig_name,
        "units": record.units,
        "comments": record.comments,
    }


def load_ecg_annotations(file_path):
    """Učitava anotacije (.atr) zapisa. Vraća None ako ih nema ili se ne mogu pročitati."""
    try:
        path = _record_path(file_path)
        annotation = wfdb.rdann(str(path), "atr")
    except Exception:
        return None

    return {
        "sample": annotation.sample,
        "symbol": annotation.symbol,
    }


def get_ecg_summary(file_path):
    """Priprema sažetak zapisa za prikaz na sučelju."""
    path = _record_path(file_path)
    record = load_ecg_record(path)

    signal = record["p_signal"]
    sample_rate = record["fs"]
    total_samples = signal.shape[0]

    return {
        "record_name": path.name,
        "channel_0_signal": signal[:, 0],
        "sample_rate": sample_rate,
        "total_samples": total_samples,
        "duration_seconds": total_samples / sample_rate,
        "channels": record["sig_name"],
        "has_annotations": load_ecg_annotations(path) is not None,
    }


if __name__ == "__main__":
    summary = get_ecg_summary(BASE_DIR / "data" / "100")

    print("Ključevi:", list(summary.keys()))
    print("Zapis:", summary["record_name"])
    print("Kanali:", summary["channels"])
    print("Frekvencija:", summary["sample_rate"], "Hz")
    print("Broj uzoraka:", summary["total_samples"])
    print(f"Trajanje: {summary['duration_seconds']:.2f} s")
    print("Oblik signala (kanal 0):", summary["channel_0_signal"].shape)
    print("Ima anotacije:", summary["has_annotations"])
