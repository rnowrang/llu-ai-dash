
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dash import Dash, dcc, html, Input, Output
import dash_daq as daq
import plotly.express as px

# ---- Load data ----
DATA_PATH = os.environ.get("DATA_PATH", "LLU Imaging AI 2025.xlsx")
df = pd.read_excel(DATA_PATH, sheet_name="Imaging AI")

# Cleanup
df.columns = df.columns.str.strip()
for c in ["Platform", "Rad Section", "Domain", "Category", "Status"]:
    df[c] = df[c].astype(str).str.strip()

# Parse dates
for c in ["Starting Date", "Ending Date"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

df["Active Flag"] = np.where(df["Status"].str.lower().str.contains("active"), "Active", "Not Active")
df["Year"] = df["Starting Date"].dt.year

# ---- App ----
app = Dash(__name__)
app.title = "LLU Imaging AI — Mini Dashboard"
server = app.server  # <-- required for Gunicorn (Render/Heroku/etc.)

def build_treemap(dff):
    return px.treemap(
        dff, path=["Domain", "Rad Section", "Category", "Name"],
        color="Rad Section", title="Treemap: Domain → Section → Category → Solution"
    )

def build_section_bar(dff):
    counts = dff.groupby("Rad Section").size().reset_index(name="Count").sort_values("Count", ascending=False)
    fig = px.bar(counts, x="Rad Section", y="Count", title="AI Solutions by Section")
    return fig

def build_platform_bar(dff):
    counts = dff.groupby("Platform").size().reset_index(name="Count").sort_values("Count", ascending=True)
    fig = px.bar(counts, x="Count", y="Platform", orientation="h", title="AI Solutions by Platform")
    return fig

def build_category_stack(dff):
    counts = dff.groupby(["Category", "Rad Section"]).size().reset_index(name="Count")
    fig = px.bar(counts, x="Category", y="Count", color="Rad Section", barmode="stack",
                 title="AI Solutions by Category (stacked by Section)")
    return fig

def build_timeline(dff):
    t = dff.copy()
    t["EndPlot"] = t["Ending Date"].fillna(pd.Timestamp(datetime.now().date()))
    t = t.dropna(subset=["Starting Date"])
    fig = px.timeline(t.sort_values("Starting Date"),
                      x_start="Starting Date", x_end="EndPlot",
                      y="Name", color="Rad Section",
                      hover_data=["Platform", "Category", "Status"],
                      title="Adoption Timeline")
    fig.update_yaxes(autorange="reversed")
    return fig

app.layout = html.Div([
    html.H1("LLU Imaging AI — Mini Dashboard"),
    html.P("Use the global switch to focus on Active solutions only, or turn it off to use the Status dropdown."),

    html.Div([
        daq.BooleanSwitch(id="active_only", on=True, color="#2c7be5"),
        html.Span(" Show only Active Solutions", style={"marginLeft": 8, "fontWeight": 600})
    ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "12px"}),

    html.Div([
        html.Div([
            html.Label("Domain"),
            dcc.Dropdown(
                options=[{"label": d, "value": d} for d in sorted(df["Domain"].unique())],
                value=None, multi=True, id="domain"
            ),
        ], style={"flex": 1, "minWidth": 220, "marginRight": 12}),

        html.Div([
            html.Label("Status (disabled when Active-only is ON)"),
            dcc.Dropdown(
                options=[{"label": s, "value": s} for s in sorted(df["Active Flag"].unique())],
                value=["Active"], multi=True, id="status"
            ),
        ], style={"flex": 1, "minWidth": 220, "marginRight": 12}),

        html.Div([
            html.Label("Section"),
            dcc.Dropdown(
                options=[{"label": s, "value": s} for s in sorted(df["Rad Section"].unique())],
                value=None, multi=True, id="section"
            ),
        ], style={"flex": 1, "minWidth": 240, "marginRight": 12}),

        html.Div([
            html.Label("Platform"),
            dcc.Dropdown(
                options=[{"label": p, "value": p} for p in sorted(df["Platform"].unique())],
                value=None, multi=True, id="platform"
            ),
        ], style={"flex": 1, "minWidth": 220}),

    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": 16}),

    dcc.Graph(id="treemap"),
    html.Div([
        html.Div([dcc.Graph(id="bar_section")], style={"flex": 1, "minWidth": 400}),
        html.Div([dcc.Graph(id="bar_platform")], style={"flex": 1, "minWidth": 400}),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px"}),
    dcc.Graph(id="cat_stack"),
    dcc.Graph(id="timeline"),
], style={"padding": "18px", "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"})


@app.callback(
    Output("treemap", "figure"),
    Output("bar_section", "figure"),
    Output("bar_platform", "figure"),
    Output("cat_stack", "figure"),
    Output("timeline", "figure"),
    Input("active_only", "on"),
    Input("domain", "value"),
    Input("status", "value"),
    Input("section", "value"),
    Input("platform", "value"),
)
def update_figs(active_only, domain, status, section, platform):
    dff = df.copy()

    if active_only:
        dff = dff[dff["Active Flag"] == "Active"]
    else:
        if status:
            dff = dff[dff["Active Flag"].isin(status)]

    if domain:
        dff = dff[dff["Domain"].isin(domain)]
    if section:
        dff = dff[dff["Rad Section"].isin(section)]
    if platform:
        dff = dff[dff["Platform"].isin(platform)]

    return (
        build_treemap(dff),
        build_section_bar(dff),
        build_platform_bar(dff),
        build_category_stack(dff),
        build_timeline(dff),
    )

if __name__ == "__main__":
    # Respect platform PORT env (Render/Spaces) and bind to 0.0.0.0
    port = int(os.environ.get("PORT", "8050"))
    app.run_server(debug=False, host="0.0.0.0", port=port)
