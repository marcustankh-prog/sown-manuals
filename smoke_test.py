"""Smoke test: load sample_daisy.json, render HTML and (try) PDF."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import Flower
from src import pdf_generator

sample = ROOT / "src" / "data" / "sample_daisy.json"
flower = Flower.model_validate_json(sample.read_text(encoding="utf-8"))

html = pdf_generator.render_html(flower, embed_css=True)
out_html = ROOT / "output" / "smoke_daisy.html"
out_html.parent.mkdir(exist_ok=True)
out_html.write_text(html, encoding="utf-8")
print(f"HTML OK: {out_html} ({len(html):,} chars)")

try:
    out_pdf = ROOT / "output" / "smoke_daisy.pdf"
    pdf_generator.to_pdf(flower, out_pdf)
    print(f"PDF OK:  {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
except Exception as e:
    print(f"PDF SKIPPED: {e}")
