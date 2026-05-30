# Beaded Flower Manual Generator

An interactive app that turns a photo of a flower into a printable beaded-flower
pattern manual styled like Henri Purnell's *Spring Bouquet*.

## Quick start (Windows)

```powershell
# 1) Create + activate venv
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install Python deps
pip install -r requirements.txt

# 3) Choose a PDF backend (one of the two below):
#
#  Option A — Playwright (recommended on Windows; no system deps)
python -m playwright install chromium
#
#  Option B — WeasyPrint (requires GTK3 runtime)
#    Download installer:
#    https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
#    After install, restart your shell so the PATH update takes effect.

# 4) Set API keys
Copy-Item .env.example .env
# edit .env and add OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 5) Run the app
streamlit run src/app.py
```

## How it works

1. Upload a flower photo.
2. The vision model drafts a `Flower` spec (name, colors, components,
   bead counts, materials, technique steps).
3. Edit any field in the form.
4. Preview the rendered HTML manual.
5. Export to PDF (A4) with the same look as the reference pattern.

## Project layout

```
src/
  app.py              Streamlit UI
  models.py           Pydantic data model
  pdf_generator.py    Jinja2 + WeasyPrint -> PDF
  ai_analyzer.py      Vision model -> Flower draft
  templates/
    manual.html.j2    Manual layout
  static/
    style.css         Print stylesheet
  data/
    sample_daisy.json Demo data extracted from the reference pattern
reference/
  TheSpringBouquetPatternbyHenriPurnellv2.pdf
  analyze_pdf.py      One-off helper used to extract layout info
output/               Generated PDFs land here
```

## Notes

- The AI draft is a *starting point*. Real bead counts, wire gauges, and
  technique steps should always be reviewed by a beader before sharing.
- Photos and copyright are yours; the reference PDF is kept locally for
  layout inspiration only.
