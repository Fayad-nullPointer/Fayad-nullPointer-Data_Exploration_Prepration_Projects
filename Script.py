# ============================================================
# NephroCare — Chronic Kidney Disease Clinical Dashboard
# ============================================================
# A professional, interactive dashboard for CKD data
# exploration, clinical analytics & XGBoost risk prediction.
# Run:  python Script.py
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# ── data ─────────────────────────────────────────────────────
df_train = pd.read_csv(os.path.join(BASE, "Training_CKD_dataset.csv"))
df_test  = pd.read_csv(os.path.join(BASE, "Testing_CKD_dataset.csv"))
df = pd.concat([df_train, df_test], ignore_index=True)

CAT_COLS = ["Diabetes", "Hypertension", "Smoking_Status", "Family_History_Kidney"]
YES_NO   = {"Yes": 1, "No": 0}
for c in CAT_COLS:
    if c in df.columns and df[c].dtype == object:
        df[c + "_enc"] = df[c].map(YES_NO).fillna(0).astype(int)

df["Pulse_Pressure"] = df["Systolic_BP"] - df["Diastolic_BP"]

FEATURE_COLS = [c for c in df_train.columns if c != "Target"]
NUMERIC_FEATURES = df[FEATURE_COLS].select_dtypes(include="number").columns.tolist()

# Prepare a fully‑numeric copy for model use
df_model = df.copy()
for c in CAT_COLS:
    if c in df_model.columns and df_model[c].dtype == object:
        df_model[c] = df_model[c].map(YES_NO).fillna(0).astype(int)

# ── model ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE, "best_model_xgb.pkl")
try:
    with open(MODEL_PATH, "rb") as fh:
        model = pickle.load(fh)
    MODEL_OK = True
    MODEL_CLASSES = list(model.classes_) if hasattr(model, "classes_") else []
except Exception:
    model, MODEL_OK, MODEL_CLASSES = None, False, []

# ── constants ────────────────────────────────────────────────
STAGES = sorted(df["Target"].dropna().unique().tolist())
STAGE_COLORS = {
    "Healthy Kidney": "#22c55e",
    "Early CKD (Stage 1)": "#84cc16",
    "Moderate CKD (Stage 2)": "#eab308",
    "Advanced CKD (Stage 3)": "#f97316",
    "Severe CKD (Stage 4)": "#ef4444",
    "End‑Stage (Stage 5)": "#991b1b",
}

def stage_color(s):
    return STAGE_COLORS.get(s, "#6366f1")

PALETTE = [stage_color(s) for s in STAGES]

# ── helper: stat cards ───────────────────────────────────────
def kpi_card(title, value, icon, color="#4361ee", delta=None):
    delta_el = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        dc = "#22c55e" if delta >= 0 else "#ef4444"
        delta_el = html.Span(f" {arrow} {abs(delta):.1f}%",
                             style={"fontSize": "13px", "color": dc, "marginLeft": "6px"})
    return html.Div([
        html.Div([
            html.Div(icon, style={"fontSize": "28px"}),
        ], style={"width": "52px", "height": "52px", "borderRadius": "14px",
                  "background": f"{color}18", "display": "flex",
                  "alignItems": "center", "justifyContent": "center", "flexShrink": 0}),
        html.Div([
            html.P(title, style={"margin": 0, "fontSize": "13px", "color": "#64748b",
                                 "fontWeight": 500, "letterSpacing": ".3px"}),
            html.H3([value, delta_el],
                     style={"margin": "2px 0 0", "fontSize": "26px", "fontWeight": 700,
                            "color": "#1e293b"}),
        ], style={"marginLeft": "14px"}),
    ], style={"display": "flex", "alignItems": "center", "background": "#fff",
              "borderRadius": "16px", "padding": "20px 24px",
              "boxShadow": "0 1px 3px rgba(0,0,0,.06)", "flex": "1",
              "minWidth": "220px"})

# ── plotly template ──────────────────────────────────────────
TMPL = go.layout.Template()
TMPL.layout = go.Layout(
    font_family="Inter, system-ui, sans-serif",
    font_color="#334155",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=["#4361ee", "#7c3aed", "#06b6d4", "#22c55e", "#eab308",
              "#ef4444", "#f97316", "#ec4899", "#14b8a6", "#6366f1"],
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
)

# ── CSS (inline) ─────────────────────────────────────────────
SIDEBAR_W = "260px"
EXTERNAL_CSS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
)

app_css = f"""
@import url('{EXTERNAL_CSS}');
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:'Inter',system-ui,sans-serif; background:#f1f5f9; color:#1e293b; }}
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-thumb {{ background:#cbd5e1; border-radius:3px; }}
.sidebar {{ position:fixed; top:0; left:0; width:{SIDEBAR_W}; height:100vh;
            background:linear-gradient(180deg,#1e293b 0%,#0f172a 100%);
            padding:28px 0; z-index:100; display:flex; flex-direction:column; }}
.sidebar-brand {{ color:#fff; font-size:20px; font-weight:800; padding:0 28px 28px;
                  letter-spacing:-.3px; border-bottom:1px solid #334155; }}
.sidebar-brand span {{ color:#60a5fa; }}
.nav-item {{ display:flex; align-items:center; gap:12px; padding:13px 28px;
             color:#94a3b8; font-size:14px; font-weight:500; cursor:pointer;
             text-decoration:none; transition:all .15s; border-left:3px solid transparent; }}
.nav-item:hover,.nav-item.active {{ color:#fff; background:#334155;
                                    border-left-color:#60a5fa; }}
.content {{ margin-left:{SIDEBAR_W}; padding:28px 32px; min-height:100vh; }}
.page-title {{ font-size:26px; font-weight:700; margin:0 0 4px; }}
.page-sub {{ font-size:14px; color:#64748b; margin:0 0 24px; }}
.cards-row {{ display:flex; gap:18px; flex-wrap:wrap; margin-bottom:24px; }}
.chart-card {{ background:#fff; border-radius:16px; padding:22px 24px;
               box-shadow:0 1px 3px rgba(0,0,0,.06); margin-bottom:20px; }}
.chart-title {{ font-size:16px; font-weight:600; margin:0 0 14px; color:#1e293b; }}
.grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
.grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }}
@media(max-width:1100px){{ .grid-2,.grid-3{{ grid-template-columns:1fr; }} }}
.tab-bar {{ display:flex; gap:6px; margin-bottom:20px; background:#fff;
            padding:6px; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
.tab-btn {{ padding:10px 22px; border-radius:10px; border:none; background:transparent;
            font-size:14px; font-weight:600; color:#64748b; cursor:pointer; transition:.15s; }}
.tab-btn:hover {{ color:#1e293b; }}
.tab-btn.active {{ background:#4361ee; color:#fff; }}
.pred-form label {{ display:block; font-size:13px; font-weight:600; color:#475569;
                    margin-bottom:4px; }}
.pred-form input,.pred-form select {{ width:100%; padding:9px 12px; border-radius:10px;
                     border:1px solid #e2e8f0; font-size:14px; margin-bottom:12px;
                     font-family:inherit; }}
.pred-btn {{ background:linear-gradient(135deg,#4361ee,#7c3aed); color:#fff;
             border:none; padding:12px 32px; border-radius:12px; font-size:15px;
             font-weight:600; cursor:pointer; width:100%; margin-top:6px; }}
.pred-btn:hover {{ opacity:.92; }}
.result-card {{ background:linear-gradient(135deg,#4361ee 0%,#7c3aed 100%);
               border-radius:16px; padding:28px; color:#fff; text-align:center; }}
.result-card h2 {{ margin:0 0 6px; font-size:28px; }}
.result-card p {{ margin:0; opacity:.85; font-size:14px; }}
.metric-box {{ background:#f8fafc; border-radius:12px; padding:16px; text-align:center; }}
.metric-box .val {{ font-size:24px; font-weight:700; color:#1e293b; }}
.metric-box .lbl {{ font-size:12px; color:#64748b; font-weight:500; margin-top:2px; }}
"""

# ════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════
app = Dash(__name__, suppress_callback_exceptions=True,
           meta_tags=[{"name": "viewport",
                       "content": "width=device-width, initial-scale=1"}])
app.title = "NephroCare · CKD Dashboard"

# Inject CSS via index_string (html.Style does not exist in Dash)
app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>''' + app_css + '''</style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

# ── sidebar ──────────────────────────────────────────────────
sidebar = html.Div([
    html.Div(["Nephro", html.Span("Care")], className="sidebar-brand"),
    html.Div(style={"height": "18px"}),
    dcc.Link([html.Span("📊"), "Overview"], href="/", className="nav-item active",
             id="nav-overview"),
    dcc.Link([html.Span("🔬"), "Clinical Analysis"], href="/analysis", className="nav-item",
             id="nav-analysis"),
    dcc.Link([html.Span("🤖"), "Risk Prediction"], href="/predict", className="nav-item",
             id="nav-predict"),
    dcc.Link([html.Span("📋"), "Data Explorer"], href="/data", className="nav-item",
             id="nav-data"),
    html.Div(style={"flex": 1}),
    html.Div([
        html.P("CKD Dashboard v1.0", style={"margin": 0, "fontSize": "12px",
                                             "color": "#475569"}),
        html.P("© 2026 Fayad — nullPointer", style={"margin": "2px 0 0", "fontSize": "11px",
                                                      "color": "#64748b"}),
    ], style={"padding": "18px 28px", "borderTop": "1px solid #334155"}),
], className="sidebar")

# ── layout ───────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    sidebar,
    html.Div(id="page-content", className="content"),
])

# ════════════════════════════════════════════════════════════
# PAGES
# ════════════════════════════════════════════════════════════

def page_overview():
    total = len(df)
    ckd_pct = round((df["Target"] != "Healthy Kidney").mean() * 100, 1)
    avg_egfr = round(df["eGFR"].mean(), 1)
    avg_cr = round(df["Serum_Creatinine"].mean(), 2)

    # -- stage distribution donut
    stage_counts = df["Target"].value_counts().reset_index()
    stage_counts.columns = ["Stage", "Count"]
    fig_donut = px.pie(stage_counts, names="Stage", values="Count",
                       hole=.55, color="Stage",
                       color_discrete_map={s: stage_color(s) for s in STAGES},
                       template=TMPL)
    fig_donut.update_traces(textposition="outside", textinfo="label+percent",
                            textfont_size=12)
    fig_donut.update_layout(showlegend=False, margin=dict(t=30, b=10, l=10, r=10),
                            height=370)

    # -- eGFR by stage
    fig_egfr = px.box(df, x="Target", y="eGFR", color="Target",
                      color_discrete_map={s: stage_color(s) for s in STAGES},
                      template=TMPL, category_orders={"Target": STAGES})
    fig_egfr.update_layout(showlegend=False, height=370,
                           xaxis_title="", yaxis_title="eGFR (mL/min/1.73m²)",
                           xaxis_tickangle=-25)

    # -- age distribution
    fig_age = px.histogram(df, x="Age", color="Target", nbins=30, barmode="stack",
                           color_discrete_map={s: stage_color(s) for s in STAGES},
                           template=TMPL, category_orders={"Target": STAGES})
    fig_age.update_layout(height=340, xaxis_title="Age (years)", yaxis_title="Count",
                          legend=dict(orientation="h", y=-0.25))

    # -- BP scatter
    fig_bp = px.scatter(df.sample(min(3000, len(df)), random_state=42),
                        x="Systolic_BP", y="Diastolic_BP", color="Target",
                        color_discrete_map={s: stage_color(s) for s in STAGES},
                        opacity=.45, template=TMPL,
                        category_orders={"Target": STAGES})
    fig_bp.update_layout(height=340, xaxis_title="Systolic BP (mmHg)",
                         yaxis_title="Diastolic BP (mmHg)",
                         legend=dict(orientation="h", y=-0.25))

    return html.Div([
        html.H1("Dashboard Overview", className="page-title"),
        html.P("Population‑level CKD analytics across training & testing cohorts",
               className="page-sub"),
        # KPI row
        html.Div([
            kpi_card("Total Patients", f"{total:,}", "🏥", "#4361ee"),
            kpi_card("CKD Prevalence", f"{ckd_pct}%", "⚠️", "#ef4444"),
            kpi_card("Mean eGFR", f"{avg_egfr}", "💧", "#06b6d4"),
            kpi_card("Mean Creatinine", f"{avg_cr} mg/dL", "🩸", "#f97316"),
        ], className="cards-row"),

        html.Div([
            # left col
            html.Div([
                html.Div([
                    html.H4("CKD Stage Distribution", className="chart-title"),
                    dcc.Graph(figure=fig_donut, config={"displayModeBar": False}),
                ], className="chart-card"),
            ]),
            # right col
            html.Div([
                html.Div([
                    html.H4("eGFR by CKD Stage", className="chart-title"),
                    dcc.Graph(figure=fig_egfr, config={"displayModeBar": False}),
                ], className="chart-card"),
            ]),
        ], className="grid-2"),

        html.Div([
            html.Div([
                html.H4("Age Distribution by CKD Stage", className="chart-title"),
                dcc.Graph(figure=fig_age, config={"displayModeBar": False}),
            ], className="chart-card"),
            html.Div([
                html.H4("Blood Pressure Landscape", className="chart-title"),
                dcc.Graph(figure=fig_bp, config={"displayModeBar": False}),
            ], className="chart-card"),
        ], className="grid-2"),
    ])


def page_analysis():
    # -- Serum Creatinine violin
    fig_cr = px.violin(df, x="Target", y="Serum_Creatinine", color="Target", box=True,
                       color_discrete_map={s: stage_color(s) for s in STAGES},
                       template=TMPL, category_orders={"Target": STAGES})
    fig_cr.update_layout(showlegend=False, height=380, xaxis_title="",
                         yaxis_title="Serum Creatinine (mg/dL)", xaxis_tickangle=-25)

    # -- BUN
    fig_bun = px.box(df, x="Target", y="Blood_Urea_Nitrogen", color="Target",
                     color_discrete_map={s: stage_color(s) for s in STAGES},
                     template=TMPL, category_orders={"Target": STAGES})
    fig_bun.update_layout(showlegend=False, height=380, xaxis_title="",
                          yaxis_title="BUN (mg/dL)", xaxis_tickangle=-25)

    # -- Pulse Pressure
    fig_pp = px.box(df, x="Target", y="Pulse_Pressure", color="Target",
                    color_discrete_map={s: stage_color(s) for s in STAGES},
                    template=TMPL, category_orders={"Target": STAGES})
    fig_pp.update_layout(showlegend=False, height=380, xaxis_title="",
                         yaxis_title="Pulse Pressure (mmHg)", xaxis_tickangle=-25)

    # -- Hemoglobin
    fig_hemo = px.histogram(df, x="Hemoglobin", color="Target", barmode="overlay",
                            nbins=40, opacity=.6,
                            color_discrete_map={s: stage_color(s) for s in STAGES},
                            template=TMPL, category_orders={"Target": STAGES})
    fig_hemo.update_layout(height=380, xaxis_title="Hemoglobin (g/dL)",
                           yaxis_title="Count", legend=dict(orientation="h", y=-0.28))

    # -- Comorbidity bar
    comor_data = []
    for cat in CAT_COLS:
        if cat in df.columns:
            for stage in STAGES:
                sub = df[df["Target"] == stage]
                if sub[cat].dtype == object:
                    pct = round((sub[cat] == "Yes").mean() * 100, 1)
                else:
                    pct = round(sub[cat].mean() * 100, 1)
                comor_data.append({"Factor": cat.replace("_", " "), "Stage": stage, "Pct": pct})
    df_comor = pd.DataFrame(comor_data)
    fig_comor = px.bar(df_comor, x="Factor", y="Pct", color="Stage", barmode="group",
                       color_discrete_map={s: stage_color(s) for s in STAGES},
                       template=TMPL, category_orders={"Stage": STAGES})
    fig_comor.update_layout(height=400, xaxis_title="", yaxis_title="Prevalence (%)",
                            legend=dict(orientation="h", y=-0.22))

    # -- Correlation heatmap (top features)
    top_feats = ["eGFR", "Serum_Creatinine", "Blood_Urea_Nitrogen", "Hemoglobin",
                 "Serum_Albumin", "Systolic_BP", "Diastolic_BP", "HbA1c",
                 "Fasting_Glucose", "Cholesterol", "Age", "BMI"]
    valid = [f for f in top_feats if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    corr = df[valid].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1, text=corr.values,
        texttemplate="%{text}", textfont_size=10,
    ))
    fig_corr.update_layout(height=520, template=TMPL,
                           margin=dict(l=100, r=20, t=30, b=100))

    return html.Div([
        html.H1("Clinical Analysis", className="page-title"),
        html.P("Deep‑dive into biomarkers, comorbidities & correlations",
               className="page-sub"),
        html.Div([
            html.Div([html.H4("Serum Creatinine by Stage", className="chart-title"),
                      dcc.Graph(figure=fig_cr, config={"displayModeBar": False})],
                     className="chart-card"),
            html.Div([html.H4("Blood Urea Nitrogen by Stage", className="chart-title"),
                      dcc.Graph(figure=fig_bun, config={"displayModeBar": False})],
                     className="chart-card"),
        ], className="grid-2"),
        html.Div([
            html.Div([html.H4("Pulse Pressure by Stage", className="chart-title"),
                      dcc.Graph(figure=fig_pp, config={"displayModeBar": False})],
                     className="chart-card"),
            html.Div([html.H4("Hemoglobin Distribution", className="chart-title"),
                      dcc.Graph(figure=fig_hemo, config={"displayModeBar": False})],
                     className="chart-card"),
        ], className="grid-2"),
        html.Div([html.H4("Comorbidity Prevalence by CKD Stage", className="chart-title"),
                  dcc.Graph(figure=fig_comor, config={"displayModeBar": False})],
                 className="chart-card"),
        html.Div([html.H4("Biomarker Correlation Matrix", className="chart-title"),
                  dcc.Graph(figure=fig_corr, config={"displayModeBar": False})],
                 className="chart-card"),
    ])


# ── Prediction page helpers ──────────────────────────────────
PRED_FIELDS = [
    ("Age", "number", 50), ("Gender", "select", ["0 — Female", "1 — Male"]),
    ("BMI", "number", 26), ("Systolic_BP", "number", 120),
    ("Diastolic_BP", "number", 80), ("Heart_Rate", "number", 75),
    ("Serum_Creatinine", "number", 1), ("Blood_Urea_Nitrogen", "number", 15),
    ("eGFR", "number", 90), ("Hemoglobin", "number", 14),
    ("HbA1c", "number", 6.5), ("Fasting_Glucose", "number", 100),
    ("Cholesterol", "number", 200), ("Serum_Albumin", "number", 4),
    ("Diabetes", "select", ["No", "Yes"]),
    ("Hypertension", "select", ["No", "Yes"]),
    ("Smoking_Status", "select", ["No", "Yes"]),
    ("Family_History_Kidney", "select", ["No", "Yes"]),
]

def _field(name, kind, default):
    lbl = html.Label(name.replace("_", " "))
    if kind == "select":
        inp = dcc.Dropdown(id=f"inp-{name}",
                           options=[{"label": v, "value": v} for v in default],
                           value=default[0],
                           clearable=False,
                           style={"borderRadius": "10px", "marginBottom": "12px"})
    else:
        inp = dcc.Input(id=f"inp-{name}", type="number", value=default,
                        style={"width": "100%", "padding": "9px 12px",
                               "borderRadius": "10px", "border": "1px solid #e2e8f0",
                               "fontSize": "14px", "marginBottom": "12px"})
    return html.Div([lbl, inp])


def page_predict():
    fields_left = [_field(*f) for f in PRED_FIELDS[:9]]
    fields_right = [_field(*f) for f in PRED_FIELDS[9:]]

    # Feature importance
    fi_fig = go.Figure()
    if MODEL_OK and hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else FEATURE_COLS
        idx = np.argsort(imp)[-15:]
        fi_fig = go.Figure(go.Bar(
            x=imp[idx], y=[names[i] for i in idx], orientation="h",
            marker_color="#4361ee",
        ))
        fi_fig.update_layout(template=TMPL, height=460, yaxis_title="",
                             xaxis_title="Importance", margin=dict(l=160))

    return html.Div([
        html.H1("Risk Prediction", className="page-title"),
        html.P("Enter patient data to predict CKD stage using our XGBoost model",
               className="page-sub"),
        html.Div([
            # form
            html.Div([
                html.Div([
                    html.H4("Patient Biomarkers", className="chart-title"),
                    html.Div([
                        html.Div(fields_left, style={"flex": 1}),
                        html.Div(fields_right, style={"flex": 1}),
                    ], style={"display": "flex", "gap": "24px"}, className="pred-form"),
                    html.Button("Run Prediction", id="btn-predict", className="pred-btn"),
                ], className="chart-card"),
            ], style={"flex": "1.2"}),
            # result
            html.Div([
                html.Div(id="prediction-result",
                         children=html.Div([
                             html.H2("Awaiting input …", style={"margin": 0, "fontSize": "22px"}),
                             html.P("Fill in patient data and click predict",
                                    style={"margin": "6px 0 0", "opacity": .7}),
                         ], className="result-card")),
                html.Div([
                    html.H4("Top Feature Importances", className="chart-title"),
                    dcc.Graph(figure=fi_fig, config={"displayModeBar": False}),
                ], className="chart-card", style={"marginTop": "20px"}),
            ], style={"flex": "0.8"}),
        ], style={"display": "flex", "gap": "24px", "alignItems": "flex-start",
                  "flexWrap": "wrap"}),
    ])


def page_data():
    cols_show = ["Target", "Age", "Gender", "BMI", "Systolic_BP", "Diastolic_BP",
                 "Serum_Creatinine", "eGFR", "Hemoglobin", "HbA1c",
                 "Diabetes", "Hypertension"]
    valid_cols = [c for c in cols_show if c in df.columns]

    tbl = dash_table.DataTable(
        id="data-table",
        columns=[{"name": c, "id": c} for c in valid_cols],
        data=df[valid_cols].head(200).to_dict("records"),
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f8fafc", "fontWeight": 600,
                      "border": "none", "fontSize": "13px", "color": "#475569"},
        style_cell={"fontSize": "13px", "padding": "10px 14px",
                    "border": "none", "borderBottom": "1px solid #f1f5f9"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"},
        ],
    )

    # descriptive statistics
    desc = df[NUMERIC_FEATURES].describe().T.round(2).reset_index()
    desc.columns = ["Feature"] + list(desc.columns[1:])
    tbl_desc = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in desc.columns],
        data=desc.to_dict("records"),
        page_size=12,
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f8fafc", "fontWeight": 600,
                      "border": "none", "fontSize": "13px", "color": "#475569"},
        style_cell={"fontSize": "13px", "padding": "10px 14px",
                    "border": "none", "borderBottom": "1px solid #f1f5f9"},
    )

    return html.Div([
        html.H1("Data Explorer", className="page-title"),
        html.P("Browse raw patient records and summary statistics", className="page-sub"),
        html.Div([
            html.H4("Patient Records (first 200 rows)", className="chart-title"),
            tbl,
        ], className="chart-card"),
        html.Div([
            html.H4("Descriptive Statistics", className="chart-title"),
            tbl_desc,
        ], className="chart-card"),
    ])


# ════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════

@app.callback(Output("page-content", "children"),
              Input("url", "pathname"))
def render_page(path):
    if path == "/analysis":
        return page_analysis()
    if path == "/predict":
        return page_predict()
    if path == "/data":
        return page_data()
    return page_overview()


# ── prediction callback ─────────────────────────────────────
_pred_inputs = [Input("btn-predict", "n_clicks")]
_pred_states = [State(f"inp-{f[0]}", "value") for f in PRED_FIELDS]

@app.callback(Output("prediction-result", "children"),
              _pred_inputs, _pred_states, prevent_initial_call=True)
def run_prediction(n_clicks, *vals):
    if not MODEL_OK:
        return html.Div([
            html.H2("Model unavailable", style={"margin": 0}),
            html.P("Could not load best_model_xgb.pkl"),
        ], className="result-card")

    # Build a single-row DataFrame with ALL features the model expects
    row = {}
    for (name, kind, _), val in zip(PRED_FIELDS, vals):
        if kind == "select":
            if name == "Gender":
                row[name] = int(str(val)[0])
            elif name in CAT_COLS:
                row[name] = YES_NO.get(val, 0)
            else:
                row[name] = val
        else:
            row[name] = float(val) if val is not None else 0.0

    # Fill missing features with training-set medians
    model_feats = (list(model.feature_names_in_)
                   if hasattr(model, "feature_names_in_") else FEATURE_COLS)
    for f in model_feats:
        if f not in row:
            if f in df_model.columns and pd.api.types.is_numeric_dtype(df_model[f]):
                row[f] = float(df_model[f].median())
            else:
                row[f] = 0

    X = pd.DataFrame([row])[model_feats]

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    # build result UI
    children = [
        html.Div([
            html.P("Predicted CKD Stage", style={"margin": 0, "fontSize": "14px", "opacity": .8}),
            html.H2(str(pred), style={"margin": "8px 0 0", "fontSize": "28px"}),
        ], className="result-card"),
    ]
    if proba is not None:
        prob_items = []
        for cls, p in sorted(zip(MODEL_CLASSES, proba), key=lambda x: -x[1]):
            prob_items.append(
                html.Div([
                    html.Div(style={"width": f"{p*100:.0f}%", "height": "8px",
                                    "borderRadius": "4px",
                                    "background": stage_color(str(cls)),
                                    "transition": "width .4s"}),
                    html.Span(f"{cls}  —  {p*100:.1f}%",
                              style={"fontSize": "12px", "color": "#475569",
                                     "marginTop": "2px", "display": "block"}),
                ], style={"marginBottom": "10px",
                          "background": "#f1f5f9", "borderRadius": "8px",
                          "padding": "8px 12px"})
            )
        children.append(html.Div([
            html.H4("Class Probabilities", className="chart-title",
                     style={"marginTop": "18px"}),
            *prob_items
        ], className="chart-card"))

    return html.Div(children)


# ════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🏥  NephroCare Dashboard starting …")
    print("   → http://127.0.0.1:8050\n")
    app.run(debug=True, port=8050)
