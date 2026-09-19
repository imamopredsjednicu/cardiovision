-- inicijalizacija baze podataka (SQLite)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name    TEXT NOT NULL,
    last_name     TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    gender        TEXT CHECK(gender IN ('M', 'Ž', 'Ostalo')),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ecg_records (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     INTEGER NOT NULL,
    file_path      TEXT NOT NULL,
    recording_date TEXT DEFAULT CURRENT_TIMESTAMP,
    bpm            INTEGER,
    status         TEXT DEFAULT 'Neobrađeno',
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);
