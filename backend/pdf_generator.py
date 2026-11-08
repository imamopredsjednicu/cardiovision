"""Generiranje PDF medicinskog nalaza EKG analize pomoću fpdf2 knjižnice."""

from datetime import datetime
from pathlib import Path

from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent.parent

# Unicode fontovi za hrvatske znakove (č, ć, đ, š, ž) - koristi se prvi pronađeni
FONT_CANDIDATES = [
    (BASE_DIR / "assets" / "fonts" / "DejaVuSans.ttf",
     BASE_DIR / "assets" / "fonts" / "DejaVuSans-Bold.ttf"),
    (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
     Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    (Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
     Path("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf")),
    (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
    (Path("/Library/Fonts/Arial Unicode.ttf"), Path("/Library/Fonts/Arial Unicode.ttf")),
]

# zamjena za ugrađeni Helvetica font koji podržava samo latin-1
ASCII_FALLBACK = str.maketrans("čćđšžČĆĐŠŽ", "ccdszCCDSZ")

FONT_NAME = "CardioFont"


def _setup_font(pdf):
    """Registrira Unicode font ako postoji na sustavu.
    Vraća (ime_fonta, funkcija_za_tekst); bez Unicode fonta tekst se transliterira.
    """
    for regular, bold in FONT_CANDIDATES:
        if regular.is_file() and bold.is_file():
            pdf.add_font(FONT_NAME, "", str(regular))
            pdf.add_font(FONT_NAME, "B", str(bold))
            return FONT_NAME, str
    return "Helvetica", lambda text: str(text).translate(ASCII_FALLBACK)


def _format_value(value, fmt="{}"):
    """Formatira vrijednost za prikaz u tablici; None prikazuje kao crticu."""
    return "-" if value is None else fmt.format(value)


def _add_table(pdf, font, text, title, rows):
    """Dodaje naslov sekcije i dvostupčanu tablicu (naziv, vrijednost)."""
    pdf.set_font(font, "B", 13)
    pdf.cell(0, 10, text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font, "", 11)
    with pdf.table(col_widths=(1, 1), text_align="LEFT", first_row_as_headings=False) as table:
        for label, value in rows:
            row = table.row()
            row.cell(text(label))
            row.cell(text(value))
    pdf.ln(6)


def generate_pdf_report(patient_data, analysis_data, graph_image_path, output_filename="nalaz.pdf"):
    """Stvara PDF nalaz s podacima o pacijentu, rezultatima analize i EKG grafikonom.
    patient_data: rječnik s ključevima first_name, last_name, date_of_birth, gender.
    analysis_data: rječnik s ključevima bpm, total_beats, arrhythmia_ratio (udio 0 - 1).
    Vraća putanju spremljene PDF datoteke.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    font, text = _setup_font(pdf)

    # zaglavlje
    pdf.set_font(font, "B", 16)
    pdf.cell(0, 10, text("CardioVision - medicinski nalaz EKG analize"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(font, "", 11)
    pdf.cell(0, 8, text(f"Datum: {datetime.now().strftime('%d.%m.%Y. %H:%M')}"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.line(pdf.l_margin, pdf.get_y() + 2, pdf.w - pdf.r_margin, pdf.get_y() + 2)
    pdf.ln(8)

    _add_table(pdf, font, text, "Podaci o pacijentu", [
        ("Ime", _format_value(patient_data.get("first_name"))),
        ("Prezime", _format_value(patient_data.get("last_name"))),
        ("Datum rođenja", _format_value(patient_data.get("date_of_birth"))),
        ("Spol", _format_value(patient_data.get("gender"))),
    ])

    ratio = analysis_data.get("arrhythmia_ratio")
    _add_table(pdf, font, text, "Rezultati analize", [
        ("Prosječni BPM", _format_value(analysis_data.get("bpm"), "{:.1f}")),
        ("Ukupno detektiranih otkucaja", _format_value(analysis_data.get("total_beats"))),
        ("Udio aritmičnih otkucaja",
         _format_value(None if ratio is None else ratio * 100, "{:.2f} %")),
    ])

    # EKG grafikon centriran po širini stranice
    if graph_image_path and Path(graph_image_path).is_file():
        pdf.set_font(font, "B", 13)
        pdf.cell(0, 10, text("EKG grafikon"), new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(graph_image_path), x="C", w=pdf.epw)
    else:
        pdf.set_font(font, "", 11)
        pdf.cell(0, 10, text("Grafikon nije dostupan."), new_x="LMARGIN", new_y="NEXT")

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return output_path


if __name__ == "__main__":
    import tempfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tmp_dir = Path(tempfile.mkdtemp(prefix="cardiovision_"))

    # privremeni grafikon sa sintetičkim EKG-om (Gaussovi "R-vrhovi" + šum)
    fs = 360
    t = np.arange(0, 5, 1 / fs)
    signal = sum(np.exp(-((t - beat) ** 2) / (2 * 0.01 ** 2)) for beat in np.arange(0.4, 5, 0.8))
    signal = signal + 0.05 * np.random.default_rng(0).standard_normal(t.size)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal, linewidth=0.8)
    ax.set_xlabel("Vrijeme (s)")
    ax.set_ylabel("Amplituda (mV)")
    ax.set_title("Testni EKG signal")
    ax.grid(alpha=0.3)
    graph_path = tmp_dir / "ekg_graf.png"
    fig.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    patient = {
        "first_name": "Ivan",
        "last_name": "Horvatić",
        "date_of_birth": "1975-04-12",
        "gender": "M",
    }
    analysis = {
        "bpm": 75.0,
        "total_beats": 2273,
        "arrhythmia_ratio": 0.0145,
    }

    pdf_path = generate_pdf_report(patient, analysis, graph_path, tmp_dir / "nalaz.pdf")
    print("PDF nalaz spremljen:", pdf_path)