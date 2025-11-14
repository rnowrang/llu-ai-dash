# LLU Imaging AI Dashboard

This Dash-based project surfaces Radiology-focused AI work through interactive treemaps, timeline plots, KPI cards, and filter controls. The app loads from the shared `LLU Imaging AI 2025.xlsx` workbook and exposes `server` so that any WSGI host can run it.

## Features

- KPI summary cards and filter summary text that update with every selection.
- Filter persistence within a session, plus a single-button “Reset filters” quick path.
- `/health` endpoint (JSON) plus structured logging for data loads and unexpected Excel/IO errors.

## Getting started

1. **Clone the repo & install deps**

```bash
git clone https://github.com/rnowrang/llu-ai-dash.git
cd llu-ai-dash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
# or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements-dev.txt
```

2. **Configure your environment**

Copy `.env.example` to `.env` (which is ignored) and adjust any values as needed.

```bash
cp .env.example .env  # Windows can use `copy`
```
The app now loads `.env` automatically at startup, so once your file contains `ONEDRIVE_DOWNLOAD_URL` the sync button and auto-download see it immediately.

3. **Run locally**

```bash
python app.py
# open http://127.0.0.1:8050 in your browser
```

### Syncing the Excel workbook

- If your master workbook lives in OneDrive, set `ONEDRIVE_DOWNLOAD_URL` to the share link (make sure the link supports `download=1`) and run:

```bash
ONEDRIVE_DOWNLOAD_URL="https://1drv.ms/..." python scripts/download_onedrive.py
# or on Windows PowerShell:
$Env:ONEDRIVE_DOWNLOAD_URL="https://1drv.ms/..."
python scripts/download_onedrive.py
```

- The script saves the file to `DATA_PATH` (defaults to `LLU Imaging AI 2025.xlsx`), so run it again whenever the sheet updates before you start the app.
- Alternately, set `ONEDRIVE_DOWNLOAD_URL` before running `python app.py` and the app will download the workbook automatically at startup (it still respects `DATA_PATH` and warns when the download fails).
- The dashboard now shows a **Sync workbook** button near the filters. Click it to re-download the workbook mid-session (the button uses the same `ONEDRIVE_DOWNLOAD_URL`), and the graphs refresh automatically after the download completes.

### Environment variables

- `DATA_PATH` – path to the Excel workbook (defaults to `LLU Imaging AI 2025.xlsx`)
- `PORT` – the port Dash listens on (defaults to `8050`)

## Observability

- `GET /health` returns JSON with service status, the timestamp when the data file was last reloaded, and any load errors.
- The app uses the standard Python logging stack (stdout) so platforms like Render, Hugging Face, or Docker can collect data-load successes/failures automatically.

## Testing

Run the test suite before pushing feature branches:

```bash
pytest
```

## Development flow

- Use short-lived feature/fix branches (`feature/`, `fix/`, `chore/` prefixes).
- Rebase `main` before starting a new branch and before opening a PR (`git pull --rebase origin main`).
- Reference `docs/dev-guide.md` for the full checklist (branching, testing, documentation, and CI).
- Keep `README_DEPLOY.md` for deployment-specific instructions suited to Render/Hugging Face Spaces.

## Continuous Integration

`GitHub Actions` runs `pytest` + `flake8` on every push and PR; see `.github/workflows/ci.yml`.

## Deployment

Follow the steps in `README_DEPLOY.md` for Render, Hugging Face, Railway, or other hosting providers.

