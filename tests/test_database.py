"""Testovi za database/db_manager.py

Svaki test radi na privremenoj bazi, pa se stvarna cardiovision.db ne dira.
"""

import sqlite3

import pytest

from database import db_manager


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Preusmjerava DB_PATH na privremenu datoteku i stvara tablice."""
    db_path = tmp_path / "test_cardiovision.db"
    monkeypatch.setattr(db_manager, "DB_PATH", db_path)
    db_manager.init_db()
    return db_path


def test_init_db_creates_tables():
    """init_db stvara tablice patients i ecg_records."""
    with db_manager.get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"patients", "ecg_records"} <= tables


def test_add_patient_and_get_all_patients():
    """Spremljeni pacijent mora biti vraćen iz baze s istim podatcima."""
    patient_id = db_manager.add_patient("Ivan", "Horvat", "1985-05-12", "M")

    patients = db_manager.get_all_patients()

    assert isinstance(patient_id, int)
    assert len(patients) == 1
    patient = patients[0]
    assert patient["id"] == patient_id
    assert patient["first_name"] == "Ivan"
    assert patient["last_name"] == "Horvat"
    assert patient["date_of_birth"] == "1985-05-12"
    assert patient["gender"] == "M"
    assert patient["created_at"] is not None


def test_get_all_patients_sorted_by_last_name():
    """Pacijenti su poredani po prezimenu pa po imenu."""
    db_manager.add_patient("Marko", "Perić", "1990-01-01", "M")
    db_manager.add_patient("Ana", "Babić", "1992-03-04", "Ž")

    last_names = [p["last_name"] for p in db_manager.get_all_patients()]

    assert last_names == ["Babić", "Perić"]


def test_add_patient_invalid_gender_raises():
    """CHECK ograničenje na stupcu gender odbija nepoznate vrijednosti."""
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.add_patient("Test", "Test", "2000-01-01", "X")


def test_add_ecg_record_and_get_patient_records():
    """Spremljeni EKG zapis mora se dohvatiti za odgovarajućeg pacijenta."""
    patient_id = db_manager.add_patient("Ivan", "Horvat", "1985-05-12", "M")
    record_id = db_manager.add_ecg_record(patient_id, "data/100", bpm=72, status="Obrađeno")

    records = db_manager.get_patient_records(patient_id)

    assert len(records) == 1
    record = records[0]
    assert record["id"] == record_id
    assert record["patient_id"] == patient_id
    assert record["file_path"] == "data/100"
    assert record["bpm"] == 72
    assert record["status"] == "Obrađeno"


def test_add_ecg_record_default_status():
    """Zapis bez zadanog statusa dobiva status 'Neobrađeno' i BPM None."""
    patient_id = db_manager.add_patient("Ana", "Babić", "1992-03-04", "Ž")
    db_manager.add_ecg_record(patient_id, "data/101")

    record = db_manager.get_patient_records(patient_id)[0]

    assert record["status"] == "Neobrađeno"
    assert record["bpm"] is None


def test_get_patient_records_only_returns_own_records():
    """Pacijent dobiva samo svoje zapise."""
    first = db_manager.add_patient("Ivan", "Horvat", "1985-05-12", "M")
    second = db_manager.add_patient("Ana", "Babić", "1992-03-04", "Ž")
    db_manager.add_ecg_record(first, "data/100")
    db_manager.add_ecg_record(second, "data/101")
    db_manager.add_ecg_record(second, "data/102")

    assert len(db_manager.get_patient_records(first)) == 1
    assert len(db_manager.get_patient_records(second)) == 2


def test_add_ecg_record_unknown_patient_raises():
    """Strani ključ ne dopušta zapis za nepostojećeg pacijenta."""
    with pytest.raises(sqlite3.IntegrityError):
        db_manager.add_ecg_record(9999, "data/100")