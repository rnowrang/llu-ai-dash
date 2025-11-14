# Developer Guide

This guide collects the practices we follow when adding features, fixing bugs, or preparing releases for the LLU Imaging AI Dashboard.

## Branching & workflow

- Start from the latest `main`: `git switch main && git pull --rebase origin main`.
- Create descriptive branches such as `feature/add-filter-switch`, `fix/typo-in-timeline`, or `chore/upgrade-deps`.
- Keep commits focused (“Add dataset filtering controls”, “Document CI pipeline”) and avoid bundling unrelated changes.
- Rebase frequently: before opening a PR, rebase your branch onto `main` so CI runs against the most recent code.
- Push the branch and open a pull request even if you are the only reviewer. The PR is the place to verify tests/linters and describe context.
- Only merge to `main` after CI passes and any blockers (comments, dependencies) are resolved.

## Environment setup

1. Python 3.11+ is required. Install dependencies with `pip install -r requirements-dev.txt`.
2. Duplicate `.env.example` to `.env` and adjust paths/ports. This repo loads `.env` automatically and reads `DATA_PATH`, `PORT`, and `ONEDRIVE_DOWNLOAD_URL` from the environment.
3. Run `python app.py` (or `gunicorn app:server`) for local smoke testing.

## Testing & linting

- Run `pytest` before pushing a branch. Tests live under `tests/` and should exercise builders/callbacks without requiring a production deployment.
- Run `flake8` to catch style issues. The CI workflow already runs it, but it’s good to run locally as well.
- For data-related changes, rerun the workbook cleanup and verify filters behave as expected.

## UI & observability

- Use the built-in `/health` endpoint for readiness checks; it reports the data refresh timestamp plus any load errors, and it is surfaced through the Flask server that Dash provides.
- Logging already emits structured messages for every data load attempt; keep those logs informative when you change the loader or add new configuration.
- Keep the filter summary text, KPI cards, and dropdown-based controls aligned when you add new filters—`filter-store` keeps the selection rally and the “Reset filters” button resets everything to the defaults defined in `default_filter_state()`.

## Data sync automation

- Before running the Dash server, download the OneDrive workbook via `scripts/download_onedrive.py` (set `ONEDRIVE_DOWNLOAD_URL` and optionally `DATA_PATH`). Run `ONEDRIVE_DOWNLOAD_URL="https://1drv.ms/..." python scripts/download_onedrive.py` whenever the workbook updates.
- When `ONEDRIVE_DOWNLOAD_URL` is defined, the app automatically fetches the workbook at startup before reading it, so your deployment just needs to expose the env var and the source file lands at `DATA_PATH` before filters/graphs render.
- The dashboard UI also exposes a **Sync workbook** button near the filters. Clicking it reruns the download (using the same `ONEDRIVE_DOWNLOAD_URL`), refreshes the cached DataFrame, and updates the charts without restarting the server.

## Pull request checklist

- [ ] Branch built off latest `main`
- [ ] Tests added/updated and passing (`pytest`)
- [ ] Style checks (`flake8`) pass
- [ ] Documentation updated (`README.md`, `README_DEPLOY.md`, `.env.example`)
- [ ] Changes described in PR description (feature, reason, testing)
- [ ] Target deployment configuration still valid

## Continuous integration

The workflow lives at `.github/workflows/ci.yml`. It runs on pushes and pull requests, installing `requirements-dev.txt`, executing `pytest`, and then running `flake8`. Keep the workflow aligned with the dependencies in `requirements-dev.txt`.

## Deployment reminders

- For hosting, keep following `README_DEPLOY.md`—Render/Hugging Face/Railway steps remain the source of truth.
- The Dash app exposes `/health` implicitly by hosting on `server` (Gunicorn/Render). Extend health checks if you add new services or APIs.
- Document any new environment variables in `.env.example` and mention them in the README.

## Next steps when adding a feature

1. Add/change code in a feature branch.
2. Update docs (`README.md`, `docs/dev-guide.md`, `README_DEPLOY.md`) with new steps or env vars.
3. Add automated tests.
4. Run `pytest` and `flake8`.
5. Push branch and open PR; leave checklist items in PR description to make reviews faster.

