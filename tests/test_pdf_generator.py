"""Testovi za backend/pdf_generator.py"""

import pytest

pytest.importorskip("fpdf")

from backend.pdf_generator import generate_pdf_report  # noqa: E402

PATIENT = {
    "first_name": "Ivan",
    "last_name": "Horvatić",
    "date_of_birth": "1975-04-12",
    "gender": "M",
}
ANALYSIS = {"bpm": 75.0, "total_beats": 2273, "arrhythmia_ratio": 0.0145}


def assert_is_pdf(path):
    assert path.is_file()
    assert path.read_bytes().startswith(b"%PDF")


def test_generate_pdf_report_without_graph(tmp_path):
    """Nepostojeći grafikon zamjenjuje se napomenom umjesto rušenja."""
    path = generate_pdf_report(PATIENT, ANALYSIS, tmp_path / "nema.png", tmp_path / "nalaz.pdf")

    assert_is_pdf(path)


@pytest.mark.parametrize("patient, analysis", [
    (None, None),
    ({}, {}),
    ({"first_name": "  "}, {"bpm": None, "total_beats": None, "arrhythmia_ratio": None}),
    ({"first_name": "Ana"}, {"bpm": float("nan"), "arrhythmia_ratio": "nepoznato"}),
    ({"first_name": "Ana"}, {"bpm": "n/a", "total_beats": 0, "arrhythmia_ratio": 0.0}),
])
def test_generate_pdf_report_missing_or_invalid_fields(tmp_path, patient, analysis):
    """Nedostajuća ili neispravna polja ne ruše izradu nalaza."""
    path = generate_pdf_report(patient, analysis, None, tmp_path / "nalaz.pdf")

    assert_is_pdf(path)


def test_generate_pdf_report_corrupt_graph(tmp_path):
    """Oštećena slika grafikona ne zaustavlja izradu nalaza."""
    graph = tmp_path / "graf.png"
    graph.write_bytes(b"ovo nije slika")

    path = generate_pdf_report(PATIENT, ANALYSIS, graph, tmp_path / "nalaz.pdf")

    assert_is_pdf(path)


def test_generate_pdf_report_creates_output_dir(tmp_path):
    """Izlazna mapa stvara se ako ne postoji."""
    path = generate_pdf_report(PATIENT, ANALYSIS, None, tmp_path / "a" / "b" / "nalaz.pdf")

    assert_is_pdf(path)
