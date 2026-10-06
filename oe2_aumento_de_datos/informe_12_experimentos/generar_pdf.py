"""
Convierte INFORME.md en INFORME.pdf (con las figuras incrustadas) usando Chrome o Edge en modo sin ventana.

Requiere: `pip install markdown` y Chrome o Edge instalados. Uso (desde la raiz del repositorio):
    python oe2_aumento_de_datos/informe_12_experimentos/generar_pdf.py
"""

import subprocess
import sys
from pathlib import Path

import markdown

CARPETA = Path(__file__).resolve().parent
MD = CARPETA / "INFORME.md"
HTML = CARPETA / "_INFORME.html"
PDF = CARPETA / "INFORME.pdf"
NAVEGADORES = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
               r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; border-bottom: 2px solid #2e7d32; padding-bottom: 6px; }
h2 { font-size: 14.5pt; margin-top: 22px; color: #1b5e20; border-bottom: 1px solid #c8e6c9; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 16px; color: #333; page-break-after: avoid; }
table { border-collapse: collapse; margin: 10px 0; width: 100%; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bdbdbd; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #e8f5e9; }
img { max-width: 100%; display: block; margin: 8px auto; page-break-inside: avoid; }
blockquote { border-left: 4px solid #2e7d32; margin: 10px 0; padding: 4px 12px; background: #f5f9f5; }
code { background: #f1f1f1; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
li { margin-bottom: 3px; }
p { margin: 6px 0; }
"""

cuerpo = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables", "sane_lists"])
HTML.write_text(f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Los 12 experimentos del OE2</title><style>{CSS}</style></head><body>{cuerpo}</body></html>", encoding="utf-8")

navegador = next((n for n in NAVEGADORES if Path(n).exists()), None)
if navegador is None:
    sys.exit("No se encontro Chrome ni Edge.")
subprocess.run([navegador, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, timeout=180)
HTML.unlink()
print("PDF generado:", PDF, f"({PDF.stat().st_size // 1024} KB)")
