
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html, no_update
import dash_daq as daq
from flask import jsonify
from pathlib import Path
from dotenv import load_dotenv

from data_sync import download_file

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

DATA_PATH = Path(os.environ.get("DATA_PATH", "LLU Imaging AI 2025.xlsx"))
ONEDRIVE_DOWNLOAD_URL = os.environ.get("ONEDRIVE_DOWNLOAD_URL")


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


@lru_cache(maxsize=1)
def load_and_prepare_data(path: str) -> pd.DataFrame:
    global DATA_REFRESHED_AT, LOAD_ERROR
    try:
        df = pd.read_excel(path, sheet_name="Imaging AI")
        df = _normalize_dataframe(df)
        DATA_REFRESHED_AT = datetime.now(timezone.utc)
        LOAD_ERROR = None
        logger.info("Loaded %d rows from %s", len(df), path)
        return df
    except Exception as exc:
        DATA_REFRESHED_AT = datetime.now(timezone.utc)
        LOAD_ERROR = str(exc)
        logger.exception("Failed to load data from %s", path)
        return _empty_dataframe()


def ensure_local_workbook() -> Path:
    if ONEDRIVE_DOWNLOAD_URL:
        try:
            logger.info("Downloading workbook from OneDrive URL")
            download_file(ONEDRIVE_DOWNLOAD_URL, DATA_PATH)
        except Exception:
            logger.exception("Failed to download workbook from OneDrive")
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
    return px.treemap(
        dff, path=["Domain", "Rad Section", "Category", "Name"],
        color="Rad Section", title=title
    )


def build_section_bar(dff: pd.DataFrame) -> px.bar:
    counts = (
        dff.groupby("Rad Section")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    return px.bar(counts, x="Rad Section", y="Count", title="AI Solutions by Section")


def build_platform_bar(dff: pd.DataFrame) -> px.bar:
    counts = (
        dff.groupby("Platform")
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=True)
    )
    return px.bar(
        counts, x="Count", y="Platform", orientation="h",
        title="AI Solutions by Platform"
    )


def build_category_stack(dff: pd.DataFrame) -> px.bar:
    counts = (
        dff.groupby(["Category", "Rad Section"])
        .size()
        .reset_index(name="Count")
    )
    return px.bar(
        counts, x="Category", y="Count", color="Rad Section",
        barmode="stack", title="AI Solutions by Category (stacked by Section)"
    )


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
    dff = filter_dataframe(df, active_only, domain, status, section, platform)
    figures = (
        build_treemap(dff),
        build_section_bar(dff),
        build_platform_bar(dff),
        build_category_stack(dff),
        build_timeline(dff),
    )
    return (*figures, build_kpi_cards(dff))


@app.callback(
    Output("sync-status", "children"),
    Output("data-refresh", "data"),
    Input("sync-workbook", "n_clicks"),
    prevent_initial_call=True,
)
def sync_workbook(n_clicks):
    if not n_clicks:
        return no_update, no_update
    if not ONEDRIVE_DOWNLOAD_URL:
        return html.Span(
            "Set ONEDRIVE_DOWNLOAD_URL to enable download",
            style={"color": "#664d03", "backgroundColor": "#fff3cd", "padding": "6px 8px", "borderRadius": "4px"},
        ), no_update
    try:
        workbook = ensure_local_workbook()
        load_and_prepare_data.cache_clear()
        refreshed = load_and_prepare_data(str(workbook))
        global df
        df = refreshed
        timestamp = datetime.now(timezone.utc).isoformat()
        return (
            html.Span(
                f"Workbook refreshed at {timestamp}",
                style={"color": "#0f5132", "backgroundColor": "#d1e7dd", "padding": "6px 8px", "borderRadius": "4px"},
            ),
            {"signal": timestamp},
        )
    except Exception as exc:
        logger.exception("Manual sync attempt failed")
        return (
            html.Span(
                f"Download failed: {exc}",
                style={"color": "#842029", "backgroundColor": "#f8d7da", "padding": "6px 8px", "borderRadius": "4px"},
            ),
            no_update,
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
