
# Deploying the LLU Imaging AI Dash app

## Can I use GitHub Pages?
**GitHub Pages only hosts static files**. A Dash app is a Python web server, so it **won’t run on GitHub Pages**.
**Two options**:
1) **Static Plotly HTML**: export `LLU_Imaging_AI_Dashboard.html` and push it to a GH Pages branch — this works on Pages.
2) **Host the live Dash app** on a free service that runs Python code (Render, Hugging Face Spaces, Railway, Fly.io, etc.).

---

## Option A — Render (free web service)
1. Create a new **Web Service** from your GitHub repo.
2. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server`
3. Ensure the file `app.py` and your data file `LLU Imaging AI 2025.xlsx` are in the repo root.
4. Add environment variable if needed:
   - `DATA_PATH` = `LLU Imaging AI 2025.xlsx`

Files provided:
- `app.py` (exposes `server` for Gunicorn)
- `requirements.txt`
- `Procfile`
- `runtime.txt` (optional)

---

## Option B — Hugging Face Spaces (free)
1. Create a new **Space** → **Docker** template.
2. Upload all files in this folder, plus your `LLU Imaging AI 2025.xlsx`.
3. Spaces will build from `Dockerfile` and run `python app.py`.
4. App will be available at the Space URL.

---

## Option C — Railway / Fly.io (free tiers)
- Similar approach: point to your GitHub repo, set build to `pip install -r requirements.txt`, and start with `gunicorn app:server` (or `python app.py` if the platform injects `PORT`).

---

## Option D — Static HTML on GitHub Pages
If you don’t need the live Dash server, you can host the static, interactive Plotly HTML:
1. Build the file (already generated earlier): `LLU_Imaging_AI_Dashboard.html`
2. Create a repo (e.g., `llu-ai-dashboard`) and push the HTML file.
3. Enable **GitHub Pages** on the repo (Settings → Pages → Source: `main` / `/ (root)`).
4. Access your dashboard at `https://<your-username>.github.io/<repo-name>/LLU_Imaging_AI_Dashboard.html`.

> Note: The static HTML will **not** have dynamic filtering beyond what Plotly provides in-figure (legend toggles, hover), but it’s great for sharing.

---

## Local dev
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python app.py
# open http://127.0.0.1:8050
```
