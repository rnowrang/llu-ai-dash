
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, no_update, clientside_callback
import dash_daq as daq
from flask import jsonify
from pathlib import Path
from dotenv import load_dotenv

from data_sync import cache_bust_url, download_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()

START_TIME = datetime.now(timezone.utc)
DATA_REFRESHED_AT: datetime | None = None
LOAD_ERROR: str | None = None

REQUIRED_COLUMNS = [
    "Domain",
    "Rad Section",
    "Category",
    "Name",
    "Platform",
    "Status",
    "Starting Date",
    "Ending Date",
]

app = Dash(__name__)
app.title = "LLU Imaging AI — Mini Dashboard"
server = app.server


@server.route("/api/sync", methods=["POST"])
def api_sync():
    """Manual sync endpoint to bypass browser extension issues."""
    try:
        logger.info("Manual sync triggered via API endpoint")
        workbook = ensure_local_workbook(raise_on_error=True)

        # Clear the cache to force reload
        if hasattr(load_and_prepare_data, '_cache'):
            load_and_prepare_data._cache.clear()
            logger.info("Cleared data cache")

        # Reload data - this will use the new file mtime as cache key
        refreshed = load_and_prepare_data(str(workbook))
        global df
        df = refreshed
        timestamp = datetime.now(timezone.utc).isoformat()
        mtime = datetime.fromtimestamp(workbook.stat().st_mtime, timezone.utc).isoformat()
        logger.info("Sync complete: %d rows loaded, file mtime: %s", len(refreshed), mtime)
        return jsonify({
            "status": "success",
            "timestamp": timestamp,
            "file_mtime": mtime,
            "rows_loaded": len(refreshed),
        })
    except Exception as exc:
        logger.exception("API sync failed")
        return jsonify({"status": "error", "message": str(exc)}), 500


DATA_PATH = Path(os.environ.get("DATA_PATH", "LLU Imaging AI 2025.xlsx"))

# Debug: Log all environment variables that might contain URLs
env_url_sources = {
    "ONEDRIVE_DOWNLOAD_URL": os.environ.get("ONEDRIVE_DOWNLOAD_URL"),
    "DATA_SOURCE_URL": os.environ.get("DATA_SOURCE_URL"),
    "DOWNLOAD_URL": os.environ.get("DOWNLOAD_URL"),
}
logger.debug(
    "Environment URL sources: %s",
    {k: (v[:50] + "..." if v and len(v) > 50 else v) for k, v in env_url_sources.items()}
)

ONEDRIVE_DOWNLOAD_URL_RAW = os.environ.get("ONEDRIVE_DOWNLOAD_URL", "").strip()
logger.info(
    "Raw ONEDRIVE_DOWNLOAD_URL from environment: %s",
    repr(ONEDRIVE_DOWNLOAD_URL_RAW[:100]) if ONEDRIVE_DOWNLOAD_URL_RAW else "None"
)


# Validate URL - check if it's a valid URL and not a placeholder
def _is_valid_url(url: str) -> bool:
    """Check if URL is valid and not a placeholder."""
    if not url:
        return False
    # Check for common placeholder patterns
    placeholder_patterns = ["<>", "…", "example.com", "your-url", "placeholder"]
    if any(pattern in url.lower() for pattern in placeholder_patterns):
        logger.warning("URL contains placeholder pattern: %s", url[:100])
        return False
    # Basic URL validation
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        is_valid = parsed.scheme in ("http", "https") and parsed.netloc
        if not is_valid:
            logger.warning("URL failed basic validation: scheme=%s, netloc=%s", parsed.scheme, parsed.netloc)
        return is_valid
    except Exception as e:
        logger.warning("URL validation exception: %s", e)
        return False


ONEDRIVE_DOWNLOAD_URL = ONEDRIVE_DOWNLOAD_URL_RAW if _is_valid_url(ONEDRIVE_DOWNLOAD_URL_RAW) else None

if ONEDRIVE_DOWNLOAD_URL_RAW and not ONEDRIVE_DOWNLOAD_URL:
    logger.warning(
        "ONEDRIVE_DOWNLOAD_URL appears to be invalid or contains placeholder text: %s. "
        "Sync from OneDrive will be disabled. Please check your .env file and system environment variables.",
        ONEDRIVE_DOWNLOAD_URL_RAW[:100] + "..." if len(ONEDRIVE_DOWNLOAD_URL_RAW) > 100 else ONEDRIVE_DOWNLOAD_URL_RAW
    )
elif ONEDRIVE_DOWNLOAD_URL:
    logger.info("ONEDRIVE_DOWNLOAD_URL is valid and set (length: %d chars)", len(ONEDRIVE_DOWNLOAD_URL))
else:
    logger.info("ONEDRIVE_DOWNLOAD_URL is not set - sync from OneDrive is disabled")


def _empty_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    df["Active Flag"] = pd.Series(dtype="object")
    df["Year"] = pd.Series(dtype="float64")
    return df


def _ensure_columns(dframe: pd.DataFrame) -> pd.DataFrame:
    for column in REQUIRED_COLUMNS:
        if column not in dframe.columns:
            dframe[column] = pd.Series(dtype="object")
    return dframe


def _normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = df.columns.str.strip()
    df = _ensure_columns(df)

    trim_columns = ["Platform", "Rad Section", "Domain", "Category", "Status", "Name"]
    for column in trim_columns:
        df[column] = df[column].astype(str).str.strip()

    for column in ["Starting Date", "Ending Date"]:
        if column in df:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    status_lower = df["Status"].fillna("").str.lower()
    df["Active Flag"] = np.where(status_lower.str.contains("active"), "Active", "Not Active")
    if "Starting Date" in df:
        df["Year"] = df["Starting Date"].dt.year
    else:
        df["Year"] = pd.Series(dtype="float64")

    return df


def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load and prepare data from Excel file. Cache is based on file mtime."""
    global DATA_REFRESHED_AT, LOAD_ERROR
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning("File does not exist: %s", path)
            return _empty_dataframe()

        # Use file modification time as part of cache key to ensure fresh data
        mtime = path_obj.stat().st_mtime
        cache_key = f"{path}:{mtime}"

        # Check if we have a cached result for this exact file version
        if not hasattr(load_and_prepare_data, '_cache'):
            load_and_prepare_data._cache = {}

        if cache_key in load_and_prepare_data._cache:
            logger.debug("Using cached data for %s (mtime: %s)", path, mtime)
            return load_and_prepare_data._cache[cache_key]

        logger.info("Loading fresh data from %s (mtime: %s)", path, mtime)
        df = pd.read_excel(path, sheet_name="Imaging AI")
        df = _normalize_dataframe(df)
        DATA_REFRESHED_AT = datetime.now(timezone.utc)
        LOAD_ERROR = None

        # Cache the result with the mtime key
        load_and_prepare_data._cache[cache_key] = df
        # Keep only the most recent cache entry (by mtime)
        if len(load_and_prepare_data._cache) > 1:
            # Remove entries with older mtimes
            current_mtime = mtime
            keys_to_remove = [
                k for k in load_and_prepare_data._cache.keys()
                if k != cache_key and k.split(':')[-1] < str(current_mtime)
            ]
            for k in keys_to_remove:
                del load_and_prepare_data._cache[k]

        logger.info("Loaded %d rows from %s", len(df), path)
        return df
    except Exception as exc:
        DATA_REFRESHED_AT = datetime.now(timezone.utc)
        LOAD_ERROR = str(exc)
        logger.exception("Failed to load data from %s", path)
        return _empty_dataframe()


def ensure_local_workbook(raise_on_error: bool = False) -> Path:
    if ONEDRIVE_DOWNLOAD_URL:
        # Double-check URL validity before attempting download
        if not _is_valid_url(ONEDRIVE_DOWNLOAD_URL):
            error_msg = f"ONEDRIVE_DOWNLOAD_URL is invalid: {ONEDRIVE_DOWNLOAD_URL[:100]}"
            logger.error(error_msg)
            if raise_on_error:
                raise ValueError(error_msg)
            return DATA_PATH

        logger.info("Local workbook path: %s", DATA_PATH.resolve())
        try:
            logger.info(
                "Using ONEDRIVE_DOWNLOAD_URL (length: %d chars): %s...",
                len(ONEDRIVE_DOWNLOAD_URL), ONEDRIVE_DOWNLOAD_URL[:50]
            )
            target_url = cache_bust_url(ONEDRIVE_DOWNLOAD_URL)
            logger.info("Downloading workbook from OneDrive URL (cache-busted)")
            download_file(target_url, DATA_PATH)
        except Exception:
            logger.exception("Failed to download workbook from OneDrive")
            if raise_on_error:
                raise
    return DATA_PATH


def default_filter_state() -> dict:
    return {
        "active_only": True,
        "domain": None,
        "status": ["Active"],
        "section": None,
        "platform": None,
    }


WORKBOOK_PATH = ensure_local_workbook()
df = load_and_prepare_data(str(WORKBOOK_PATH))


def filter_dataframe(
    dataset: pd.DataFrame,
    active_only: bool,
    domain: list | None,
    status: list | None,
    section: list | None,
    platform: list | None,
) -> pd.DataFrame:
    dff = dataset.copy()

    if active_only:
        dff = dff[dff["Active Flag"] == "Active"]
    elif status:
        dff = dff[dff["Active Flag"].isin(status)]

    if domain:
        dff = dff[dff["Domain"].isin(domain)]
    if section:
        dff = dff[dff["Rad Section"].isin(section)]
    if platform:
        dff = dff[dff["Platform"].isin(platform)]

    return dff


def build_treemap(dff: pd.DataFrame) -> px.treemap:
    title = "Treemap: Domain → Section → Category → Solution"
    fig = px.treemap(
        dff, path=["Domain", "Rad Section", "Category", "Name"],
        color="Rad Section", title=title
    )
    return _with_mode_default(fig)


def build_section_bar(dff: pd.DataFrame) -> px.bar:
    counts = (
        dff.groupby("Rad Section")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    return _with_mode_default(px.bar(counts, x="Rad Section", y="Count", title="AI Solutions by Section"))


def build_platform_bar(dff: pd.DataFrame) -> px.bar:
    counts = (
        dff.groupby("Platform")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=True)
    )
    fig = px.bar(
        counts, x="Count", y="Platform", orientation="h",
        title="AI Solutions by Platform"
    )
    return _with_mode_default(fig)


def build_category_stack(dff: pd.DataFrame) -> px.bar:
    counts = (
        dff.groupby(["Category", "Rad Section"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        counts, x="Category", y="Count", color="Rad Section",
        barmode="stack", title="AI Solutions by Category (stacked by Section)"
    )
    return _with_mode_default(fig)


def build_timeline(dff: pd.DataFrame) -> px.timeline:
    if dff.empty or "Starting Date" not in dff:
        return px.timeline(
            pd.DataFrame({"Name": [], "Starting Date": [], "Ending Date": []}),
            x_start="Starting Date",
            x_end="Ending Date",
            y="Name",
            title="Adoption Timeline (no data)",
        )

    t = dff.copy()
    end_fill = pd.Timestamp(datetime.now(timezone.utc).date())
    t["EndPlot"] = t["Ending Date"].fillna(end_fill)
    t = t.dropna(subset=["Starting Date"])
    fig = px.timeline(
        t.sort_values("Starting Date"),
        x_start="Starting Date", x_end="EndPlot",
        y="Name", color="Rad Section",
        hover_data=["Platform", "Category", "Status"],
        title="Adoption Timeline"
    )
    fig.update_yaxes(autorange="reversed")
    return _with_mode_default(fig)


def _with_mode_default(fig):
    """Ensure traces that support mode have it set to prevent Graph.react.js errors.

    Note: Some trace types (like Treemap) don't support mode, so Graph.react.js
    may still error on hover for those. This is a Plotly/Dash limitation.
    """
    for i, trace in enumerate(getattr(fig, "data", [])):
        if trace is not None:
            mode = getattr(trace, "mode", None)
            if mode is None:
                try:
                    setattr(trace, "mode", "")
                except (ValueError, AttributeError):
                    pass
    return fig


def build_kpi_cards(dff: pd.DataFrame) -> html.Div:
    total = len(dff)
    active = int((dff["Active Flag"] == "Active").sum())
    active_pct = round((active / total) * 100, 1) if total else 0.0
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    upcoming = int(
        dff["Starting Date"].ge(today).sum() if "Starting Date" in dff else 0
    )

    cards = [
        {
            "label": "Tracked solutions",
            "value": total,
            "note": "AI solutions in scope",
        },
        {
            "label": "Active solutions",
            "value": f"{active_pct}%",
            "note": f"{active} / {total}",
        },
        {
            "label": "Upcoming launches",
            "value": upcoming,
            "note": "Starting this year or later",
        },
    ]

    card_elements = []
    for card in cards:
        card_elements.append(
            html.Div(
                [
                    html.Div(card["label"], style={"color": "#6c757d", "fontSize": "0.85rem"}),
                    html.Div(card["value"], style={"fontSize": "1.8rem", "fontWeight": 600}),
                    html.Div(card["note"], style={"color": "#6c757d", "fontSize": "0.8rem"}),
                ],
                style={
                    "flex": "1",
                    "minWidth": "180px",
                    "padding": "12px 16px",
                    "border": "1px solid #e3e3e3",
                    "borderRadius": "10px",
                    "backgroundColor": "#ffffff",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                },
            )
        )

    return html.Div(
        card_elements,
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "16px"},
    )


def render_filter_summary(filter_state: dict | None) -> html.Span:
    state = filter_state or default_filter_state()
    parts = []

    if state.get("active_only"):
        parts.append("Active solutions only")
    else:
        statuses = state.get("status") or []
        status_label = ", ".join(statuses) if statuses else "Any status"
        parts.append(f"Status: {status_label}")

    for key, label in (("domain", "Domain"), ("section", "Section"), ("platform", "Platform")):
        values = state.get(key)
        if values:
            if isinstance(values, list):
                parts.append(f"{label}: {', '.join(sorted(set(values)))}")
            else:
                parts.append(f"{label}: {values}")

    summary = " | ".join(parts) if parts else "No filters applied"
    return html.Span(summary, style={"fontSize": "0.9rem", "color": "#333"})


def render_error_banner() -> html.Div | None:
    if not LOAD_ERROR:
        return None
    return html.Div(
        [
            html.Strong("Data load issue: ", style={"marginRight": "4px"}),
            html.Span(LOAD_ERROR),
        ],
        style={
            "border": "1px solid #f5c6cb",
            "backgroundColor": "#f8d7da",
            "padding": "10px 14px",
            "borderRadius": "6px",
            "color": "#721c24",
            "marginBottom": "16px",
        },
    )


app.layout = html.Div(
    [
        dcc.Store(
            id="filter-store",
            storage_type="session",
            data=default_filter_state(),
        ),
        dcc.Store(id="data-refresh", data={"signal": None}),
        html.Header(
            [
                html.H1("LLU Imaging AI — Mini Dashboard"),
                html.P(
                    "Toggle filters, explore KPIs, and get decision-ready insights. "
                    "The UI remembers your selections during the session.",
                    style={"marginTop": "4px", "color": "#555"},
                ),
            ]
        ),
        render_error_banner(),
        html.Div(
            [
                html.Div(
                    [
                        daq.BooleanSwitch(
                            id="active_only",
                            on=True,
                            color="#2c7be5",
                        ),
                        html.Span(
                            " Show only Active Solutions",
                            style={"marginLeft": 8, "fontWeight": 600},
                        ),
                        html.Button(
                            "Reset filters",
                            id="reset-filters",
                            n_clicks=0,
                            style={
                                "marginLeft": "auto",
                                "padding": "8px 14px",
                                "borderRadius": "6px",
                                "border": "1px solid #ccc",
                                "backgroundColor": "#fff",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "12px",
                        "marginBottom": "16px",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Sync workbook",
                            id="sync-workbook",
                            n_clicks=0,
                            style={
                                "padding": "8px 16px",
                                "borderRadius": "6px",
                                "border": "1px solid #2c7be5",
                                "backgroundColor": "#2c7be5",
                                "color": "#fff",
                                "cursor": "pointer",
                                "fontWeight": 600,
                            },
                        ),
                        html.Div(
                            id="sync-status",
                            style={"marginLeft": "12px", "alignSelf": "center"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginBottom": "16px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Domain"),
                                dcc.Dropdown(
                                    options=[
                                        {"label": d, "value": d}
                                        for d in sorted(df["Domain"].dropna().unique())
                                    ],
                                    value=None,
                                    multi=True,
                                    id="domain",
                                ),
                            ],
                            style={"flex": 1, "minWidth": 220, "marginRight": 12},
                        ),
                        html.Div(
                            [
                                html.Label("Status (disabled when Active-only is ON)"),
                                dcc.Dropdown(
                                    options=[
                                        {"label": s, "value": s}
                                        for s in sorted(df["Active Flag"].dropna().unique())
                                    ],
                                    value=["Active"],
                                    multi=True,
                                    id="status",
                                ),
                            ],
                            style={"flex": 1, "minWidth": 220, "marginRight": 12},
                        ),
                        html.Div(
                            [
                                html.Label("Section"),
                                dcc.Dropdown(
                                    options=[
                                        {"label": s, "value": s}
                                        for s in sorted(df["Rad Section"].dropna().unique())
                                    ],
                                    value=None,
                                    multi=True,
                                    id="section",
                                ),
                            ],
                            style={"flex": 1, "minWidth": 240, "marginRight": 12},
                        ),
                        html.Div(
                            [
                                html.Label("Platform"),
                                dcc.Dropdown(
                                    options=[
                                        {"label": p, "value": p}
                                        for p in sorted(df["Platform"].dropna().unique())
                                    ],
                                    value=None,
                                    multi=True,
                                    id="platform",
                                ),
                            ],
                            style={"flex": 1, "minWidth": 220},
                        ),
                    ],
                    style={"display": "flex", "flexWrap": "wrap"},
                ),
            ],
            style={"marginBottom": "16px"},
        ),
        html.Div(
            id="filter-summary",
            children=render_filter_summary(default_filter_state()),
            style={"padding": "8px 0", "borderBottom": "1px solid #e3e3e3"},
        ),
        html.Div(
            id="kpi-cards",
            children=build_kpi_cards(df),
        ),
        html.Div(
            [
                dcc.Graph(id="treemap"),
                html.Div(
                    [
                        html.Div([dcc.Graph(id="bar_section")], style={"flex": 1, "minWidth": 360}),
                        html.Div([dcc.Graph(id="bar_platform")], style={"flex": 1, "minWidth": 360}),
                    ],
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "marginTop": "18px",
                        "flexWrap": "wrap",
                    },
                ),
                dcc.Graph(id="cat_stack"),
                dcc.Graph(id="timeline"),
            ],
            style={"marginTop": "24px"},
        ),
    ],
    style={"padding": "18px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"},
)


@app.callback(
    Output("treemap", "figure"),
    Output("bar_section", "figure"),
    Output("bar_platform", "figure"),
    Output("cat_stack", "figure"),
    Output("timeline", "figure"),
    Output("kpi-cards", "children"),
    Input("active_only", "on"),
    Input("domain", "value"),
    Input("status", "value"),
    Input("section", "value"),
    Input("platform", "value"),
    Input("data-refresh", "data"),
)
def update_figs(active_only, domain, status, section, platform, _):
    # Reload data fresh to ensure we're using the latest file (cache handles mtime-based caching)
    global df
    current_df = load_and_prepare_data(str(WORKBOOK_PATH))
    # Update global df if row count changed (indicates file was updated)
    old_len = len(df)
    if len(current_df) != old_len:
        df = current_df
        logger.info("Data updated in update_figs: %d rows (was %d)", len(df), old_len)
    else:
        df = current_df  # Still update to ensure we have the latest cached version
    dff = filter_dataframe(df, active_only, domain, status, section, platform)
    figures = (
        build_treemap(dff),
        build_section_bar(dff),
        build_platform_bar(dff),
        build_category_stack(dff),
        build_timeline(dff),
    )
    return (*figures, build_kpi_cards(dff))


# Setup direct event listener on sync button to bypass Dash callbacks and avoid extension interference
clientside_callback(
    """
    function(_) {
        // Set up direct event listener on the sync button
        const syncButton = document.getElementById('sync-workbook');
        const statusDiv = document.getElementById('sync-status');

        if (!syncButton || syncButton.dataset.listenerAttached === 'true') {
            return window.dash_clientside.no_update;
        }

        syncButton.dataset.listenerAttached = 'true';

        syncButton.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();

            console.log('Sync button clicked (direct listener)');

            const syncingStyle = 'color: #856404; background-color: #fff3cd; padding: 6px 8px; border-radius: 4px;';
            const successStyle = 'color: #0f5132; background-color: #d1e7dd; padding: 6px 8px; border-radius: 4px;';
            const errorStyle = 'color: #842029; background-color: #f8d7da; padding: 6px 8px; border-radius: 4px;';

            if (statusDiv) {
                statusDiv.innerHTML = '<span style="' + syncingStyle + '">Syncing...</span>';
            }

            // Disable button during sync
            syncButton.disabled = true;
            syncButton.style.opacity = '0.6';
            syncButton.style.cursor = 'not-allowed';

            fetch('/api/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                cache: 'no-cache'
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('HTTP ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                console.log('Sync response:', data);
                if (data.status === 'success') {
                    const timestamp = new Date(data.timestamp).toLocaleString();
                    const msg = 'Workbook refreshed at ' + timestamp + ' (local file mtime ' + data.file_mtime + ')';
                    if (statusDiv) {
                        statusDiv.innerHTML = '<span style="' + successStyle + '">' + msg + '</span>';
                    }
                    // Reload after short delay
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                } else {
                    throw new Error(data.message || 'Sync failed');
                }
            })
            .catch(error => {
                console.error('Sync error:', error);
                const msg = 'Download failed: ' + (error.message || 'Unknown error');
                if (statusDiv) {
                    statusDiv.innerHTML = '<span style="' + errorStyle + '">' + msg + '</span>';
                }
                // Re-enable button on error
                syncButton.disabled = false;
                syncButton.style.opacity = '1';
                syncButton.style.cursor = 'pointer';
            });
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output("sync-status", "children"),
    Input("sync-workbook", "id"),  # Trigger on component mount
    prevent_initial_call=False,  # Run on initial load to attach listener
)


@app.callback(
    Output("filter-store", "data"),
    Input("active_only", "on"),
    Input("domain", "value"),
    Input("status", "value"),
    Input("section", "value"),
    Input("platform", "value"),
)
def cache_filters(active_only, domain, status, section, platform):
    return {
        "active_only": active_only,
        "domain": domain,
        "status": status,
        "section": section,
        "platform": platform,
    }


@app.callback(
    Output("filter-summary", "children"),
    Input("filter-store", "data"),
)
def update_filter_summary(filter_state):
    return render_filter_summary(filter_state)


@app.callback(
    Output("domain", "value"),
    Output("status", "value"),
    Output("section", "value"),
    Output("platform", "value"),
    Input("reset-filters", "n_clicks"),
)
def reset_filters(n_clicks):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    return None, ["Active"], None, None


@server.route("/health")
def health():
    payload = {
        "status": "error" if LOAD_ERROR else "ok",
        "started_at": START_TIME.isoformat(),
        "data_last_refresh": DATA_REFRESHED_AT.isoformat() if DATA_REFRESHED_AT else None,
        "data_error": LOAD_ERROR,
    }
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run_server(debug=False, host="0.0.0.0", port=port)
