"""Rad s bazom podataka."""

import sqlite3
from pathlib import Path

# korijenska mapa projekta
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "cardiovision.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

def get_connection():
    """Otvara konekciju na bazu s podrškom za strane ključeve.
    Redovi se vraćaju kao sqlite3.Row, pa im se može pristupati po nazivu stupca.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Čita database/schema.sql i stvara tablice ako još ne postoje."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection() as conn:
        conn.executescript(schema_sql)

def add_patient(first_name, last_name, date_of_birth, gender):
    """Unosi novog pacijenta i vraća njegov ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO patients (first_name, last_name, date_of_birth, gender)
            VALUES (?, ?, ?, ?)
            """,
            (first_name, last_name, date_of_birth, gender),
        )
        return cursor.lastrowid

def get_all_patients():
    """Vraća listu svih pacijenata poredanih po prezimenu i imenu."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, first_name, last_name, date_of_birth, gender, created_at
            FROM patients
            ORDER BY last_name, first_name
            """
        )
        return cursor.fetchall()

def add_ecg_record(patient_id, file_path, bpm=None, status="Neobrađeno"):
    """Unosi novi EKG zapis povezan s pacijentom i vraća ID zapisa."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ecg_records (patient_id, file_path, bpm, status)
            VALUES (?, ?, ?, ?)
            """,
            (patient_id, str(file_path), bpm, status),
        )
        return cursor.lastrowid

def get_patient_records(patient_id):
    """Vraća sve EKG zapise određenog pacijenta od najnovijeg do najstarijeg."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, patient_id, file_path, recording_date, bpm, status
            FROM ecg_records
            WHERE patient_id = ?
            ORDER BY recording_date DESC, id DESC
            """,
            (patient_id,),
        )
        return cursor.fetchall()

if __name__ == "__main__":
    init_db()
    print(f"Baza inicijalizirana: {DB_PATH}")
    patient_id = add_patient("Ivan", "Horvat", "1985-05-12", "M")
    print(f"Dodan testni pacijent s ID-jem: {patient_id}")
    print("\nSvi pacijenti u bazi:")
    for patient in get_all_patients():
        print(
            f"  [{patient['id']}] {patient['first_name']} {patient['last_name']} | "
            f"rođen: {patient['date_of_birth']} | spol: {patient['gender']} | "
            f"unesen: {patient['created_at']}"
        )