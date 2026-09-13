CardioVision

CardioVision je desktop aplikacija u Pythonu namijenjena analizi EKG signala, ranom otkrivanju srčanih aritmija i upravljanju kartonima pacijenata. Projekt je razvijen za Natjecanje iz informatike (kategorija Razvoj softvera).

Tehnološki stog (Tech Stack)
Frontend: CustomTkinter, Matplotlib
Backend: SciPy, NumPy, Pandas, scikit-learn
Baza podataka: SQLite (sqlite3)
Izvješća i testiranje: fpdf2, pytest
Izvor podataka: PhysioNet (WFDB format: .dat, .hea, .atr)

Tim
Luka Cerovečki – backend i algoritmi
Emil Cigula – frontend i UI/UX

API format prijenosa podataka (Backend <-> Frontend)
Backend šalje podatke sučelju u obliku Python rječnika (dict):

{
    "status": "success",          status analize
    "message": "OK",              poruka o grešci
    "bpm": 75,                    izračunati prosječni BPM
    "signal_raw": [...],          sirovi EKG signal
    "signal_filtered": [...],     filtrirani EKG signal
    "r_peaks": [120, 450, 780],   indeksi uzoraka gdje se nalaze R-vrhovi
    "arrhythmias": [450]          indeksi uzoraka gdje je detektirana aritmija
}