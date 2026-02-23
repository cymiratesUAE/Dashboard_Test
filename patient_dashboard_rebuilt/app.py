import io
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Patient Insights Dashboard",
    page_icon="🩺",
    layout="wide",
)

# ----------------------------
# Styling (white theme + blue/green accents + glowing tabs)
# ----------------------------
ACCENT_BLUE = "#1E88E5"
ACCENT_GREEN = "#00C853"

CUSTOM_CSS = f'''
<style>
:root {{
  --accent-blue: {ACCENT_BLUE};
  --accent-green: {ACCENT_GREEN};
  --card: rgba(255,255,255,0.92);
  --shadow: 0 10px 25px rgba(0,0,0,0.06);
}}

.stApp {{
  background:
    radial-gradient(1200px 600px at 10% 5%, rgba(30,136,229,0.10), transparent 55%),
    radial-gradient(1000px 520px at 90% 0%, rgba(0,200,83,0.10), transparent 55%),
    #ffffff;
}}

section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(30,136,229,0.06), rgba(0,200,83,0.05));
  border-right: 1px solid rgba(0,0,0,0.06);
}}

.block-container {{
  padding-top: 1.2rem;
}}

div[data-testid="stMetric"] {{
  background: var(--card);
  box-shadow: var(--shadow);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 10px 14px;
}}

.glow-card {{
  background: var(--card);
  box-shadow: var(--shadow);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 18px;
  padding: 14px 16px;
}}

h1, h2, h3 {{
  letter-spacing: -0.02em;
}}

div[role="tablist"] button {{
  border-radius: 999px !important;
  margin-right: .4rem !important;
  padding: .55rem .9rem !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  background: rgba(255,255,255,0.85) !important;
}}

div[role="tablist"] button[aria-selected="true"] {{
  border: 1px solid rgba(30,136,229,0.35) !important;
  box-shadow: 0 0 0 2px rgba(30,136,229,0.14), 0 0 18px rgba(0,200,83,0.12) !important;
  background: linear-gradient(90deg, rgba(30,136,229,0.10), rgba(0,200,83,0.09)) !important;
}}

.small-note {{
  opacity: .78;
  font-size: 0.92rem;
}}

.badge {{
  display: inline-block;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: rgba(30,136,229,0.10);
  border: 1px solid rgba(30,136,229,0.18);
}}
</style>
'''
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------
# Paths (robust for local + Streamlit Cloud)
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = APP_DIR / "data" / "patients.csv"

# ----------------------------
# Helpers
# ----------------------------
def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date

def _ensure_month(df: pd.DataFrame, month_label: str | None) -> pd.DataFrame:
    if month_label and month_label.strip():
        df = df.copy()
        df["MONTH"] = month_label.strip()
    elif "MONTH" not in df.columns:
        df = df.copy()
        df["MONTH"] = "Base"
    return df

def _compute_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "BIRTHDATE" in df.columns:
        b = _parse_date(df["BIRTHDATE"])
        today = date.today()
        age = pd.Series(
            [
                (today.year - x.year - ((today.month, today.day) < (x.month, x.day))) if pd.notnull(x) else np.nan
                for x in b
            ]
        )
        df["AGE"] = age
    return df

def _compute_alive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "DEATHDATE" in df.columns:
        dd = _parse_date(df["DEATHDATE"])
        df["ALIVE"] = dd.isna()
        df["DECEASED"] = ~df["ALIVE"]
    else:
        df["ALIVE"] = True
        df["DECEASED"] = False
    return df

def _age_band(age: float) -> str:
    if pd.isna(age):
        return "Unknown"
    bins = [(0, 17), (18, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 200)]
    for lo, hi in bins:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "Unknown"

def _standardize(df: pd.DataFrame, month_label: str | None) -> pd.DataFrame:
    df = df.copy()

    # normalize common lowercase variants
    if "birthdate" in df.columns and "BIRTHDATE" not in df.columns:
        df.rename(columns={"birthdate": "BIRTHDATE"}, inplace=True)
    if "deathdate" in df.columns and "DEATHDATE" not in df.columns:
        df.rename(columns={"deathdate": "DEATHDATE"}, inplace=True)

    df = _ensure_month(df, month_label)
    df = _compute_age(df)
    df = _compute_alive(df)

    if "AGE" in df.columns:
        df["AGE_BAND"] = df["AGE"].apply(_age_band)

    return df

@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(f"Default dataset not found at: {DEFAULT_CSV}")
    return pd.read_csv(DEFAULT_CSV)

def try_read_uploaded(file) -> pd.DataFrame:
    raw = file.getvalue()
    return pd.read_csv(io.BytesIO(raw))

def merge_datasets(existing: pd.DataFrame, new_df: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    df = pd.concat([existing, new_df], ignore_index=True)
    if dedupe and "Id" in df.columns and "MONTH" in df.columns:
        df = df.drop_duplicates(subset=["Id", "MONTH"], keep="last")
    elif dedupe:
        df = df.drop_duplicates(keep="last")
    return df

def filter_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    for col, val in filters.items():
        if val is None or col not in out.columns:
            continue
        if isinstance(val, tuple) and len(val) == 2:
            lo, hi = val
            out = out[(out[col].fillna(lo) >= lo) & (out[col].fillna(hi) <= hi)]
        elif isinstance(val, list):
            if len(val) == 0:
                continue
            out = out[out[col].astype(str).isin([str(x) for x in val])]
        else:
            out = out[out[col].astype(str) == str(val)]
    return out

def percent(n, d) -> float:
    return 0.0 if d == 0 else 100.0 * (n / d)

# ----------------------------
# Session data init
# ----------------------------
if "data" not in st.session_state:
    st.session_state.data = _standardize(load_base_data(), month_label="Base")

df_all = st.session_state.data

# ----------------------------
# Sidebar: upload + filters
# ----------------------------
st.sidebar.markdown("## ⚙️ Controls")
st.sidebar.caption("Upload a new month’s CSV and the dashboard updates immediately.")

upload = st.sidebar.file_uploader("Upload new dataset (CSV)", type=["csv"])
month_label = st.sidebar.text_input("Month label (optional)", placeholder="e.g., 2026-01")
dedupe = st.sidebar.checkbox("De-dupe by Id + Month", value=True)

if upload is not None:
    try:
        new_df = try_read_uploaded(upload)
        new_df = _standardize(new_df, month_label=month_label if month_label.strip() else "Uploaded")
        st.session_state.data = merge_datasets(st.session_state.data, new_df, dedupe=dedupe)
        df_all = st.session_state.data
        st.sidebar.success(f"Loaded {len(new_df):,} rows. Combined: {len(df_all):,} rows.")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔎 Filters")

def _multi(colname, label):
    if colname not in df_all.columns:
        return []
    opts = sorted([x for x in df_all[colname].dropna().astype(str).unique().tolist()])
    return st.sidebar.multiselect(label, opts, default=[])

filters = {}
for col, label in [
    ("GENDER", "Gender"),
    ("RACE", "Race"),
    ("ETHNICITY", "Ethnicity"),
    ("MARITAL", "Marital"),
    ("STATE", "State"),
    ("CITY", "City"),
    ("MONTH", "Month"),
]:
    filters[col] = _multi(col, label)

if "ALIVE" in df_all.columns:
    alive_choice = st.sidebar.selectbox("Status", ["All", "Alive", "Deceased"], index=0)
    if alive_choice == "Alive":
        filters["ALIVE"] = ["True"]
    elif alive_choice == "Deceased":
        filters["ALIVE"] = ["False"]

if "AGE" in df_all.columns and df_all["AGE"].notna().any():
    age_min, age_max = int(np.nanmin(df_all["AGE"])), int(np.nanmax(df_all["AGE"]))
    filters["AGE"] = st.sidebar.slider(
        "Age range",
        min_value=max(0, age_min),
        max_value=max(0, age_max),
        value=(max(0, age_min), max(0, age_max)),
    )

df = filter_df(df_all, filters)

st.sidebar.markdown("---")
if st.sidebar.button("♻️ Reset to base dataset"):
    st.session_state.data = _standardize(load_base_data(), month_label="Base")
    st.rerun()

# ----------------------------
# Header + KPIs
# ----------------------------
st.title("Patient Insights Dashboard")
st.markdown(
    '<span class="badge">White theme</span> &nbsp; <span class="badge">Blue/Green accents</span> &nbsp; <span class="badge">Upload new month</span>',
    unsafe_allow_html=True,
)
st.markdown('<div class="small-note">Tip: use the sidebar filters to slice cohorts, then download the filtered view.</div>', unsafe_allow_html=True)

total = len(df)
deceased = int(df["DECEASED"].sum()) if "DECEASED" in df.columns else 0
alive = int(df["ALIVE"].sum()) if "ALIVE" in df.columns else total
deceased_rate = percent(deceased, total)
distinct_months = df["MONTH"].nunique() if "MONTH" in df.columns else 1

k1, k2, k3, k4 = st.columns(4)
k1.metric("Patients (filtered)", f"{total:,}")
k2.metric("Alive", f"{alive:,}")
k3.metric("Deceased", f"{deceased:,}", f"{deceased_rate:.1f}%")
k4.metric("Months in view", f"{distinct_months:,}")

dl1, dl2 = st.columns([1, 1])
with dl1:
    st.download_button(
        "⬇️ Download filtered CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="patients_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "⬇️ Download full combined CSV",
        data=df_all.to_csv(index=False).encode("utf-8"),
        file_name="patients_combined.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_cohorts, tab_geo, tab_data = st.tabs(["📊 Overview", "🧬 Cohorts", "🗺️ Geo", "🧾 Data"])

with tab_overview:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("Population composition")
    c1, c2 = st.columns(2)

    with c1:
        if "GENDER" in df.columns:
            fig = px.pie(df, names="GENDER", title="Gender distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No GENDER column found.")

    with c2:
        cat = "RACE" if "RACE" in df.columns else ("ETHNICITY" if "ETHNICITY" in df.columns else None)
        if cat:
            tmp = df[cat].astype(str).value_counts().reset_index()
            tmp.columns = [cat, "count"]
            fig = px.bar(tmp, x=cat, y="count", title=f"{cat} counts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No RACE/ETHNICITY column found.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("Outcome signal (deceased rate) by segment")
    segment_options = [c for c in ["GENDER", "RACE", "ETHNICITY", "MARITAL", "STATE", "CITY", "MONTH", "AGE_BAND"] if c in df.columns]
    if segment_options and "DECEASED" in df.columns:
        seg = st.selectbox("Choose segment", segment_options)
        g = df.groupby(seg, dropna=False).agg(patients=("Id", "count") if "Id" in df.columns else ("MONTH", "count"), deceased=("DECEASED", "sum")).reset_index()
        g["deceased_rate_%"] = (g["deceased"] / g["patients"]).replace([np.inf, np.nan], 0) * 100
        g = g.sort_values("patients", ascending=False).head(25)
        fig = px.bar(g, x=seg, y="deceased_rate_%", hover_data=["patients", "deceased"], title="Deceased rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher bars mean higher deceased rate within that segment slice.")
    else:
        st.info("Segment analysis requires DECEASED and at least one segment column.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_cohorts:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("Cohort heatmap: deceased rate by Age Band × Segment")
    segment_options = [c for c in ["RACE", "ETHNICITY", "GENDER", "MARITAL", "STATE", "MONTH"] if c in df.columns]
    if "AGE_BAND" in df.columns and "DECEASED" in df.columns and segment_options:
        segment = st.selectbox("Segment for columns", segment_options, key="heat_seg")
        pivot = df.pivot_table(index="AGE_BAND", columns=segment, values="DECEASED", aggfunc="mean", fill_value=0) * 100.0
        order = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-200", "Unknown"]
        pivot = pivot.reindex([x for x in order if x in pivot.index])

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                colorbar=dict(title="Deceased %"),
                hovertemplate="Age band: %{y}<br>" + segment + ": %{x}<br>Deceased rate: %{z:.1f}%<extra></extra>",
            )
        )
        fig.update_layout(title="Deceased rate heatmap (%)", height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Useful for spotting risk pockets across cohorts.")
    else:
        st.info("Heatmap requires AGE_BAND, DECEASED, and at least one segment column.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("Quick narrative")
    notes = []
    if "AGE" in df.columns and df["AGE"].notna().any():
        notes.append(f"Median age: **{np.nanmedian(df['AGE']):.0f}**")
    notes.append(f"Deceased rate: **{deceased_rate:.1f}%**")
    if "STATE" in df.columns and len(df) > 0:
        notes.append(f"Top state: **{df['STATE'].astype(str).value_counts().index[0]}**")
    if "CITY" in df.columns and len(df) > 0:
        notes.append(f"Top city: **{df['CITY'].astype(str).value_counts().index[0]}**")
    st.write(" • ".join(notes))
    st.markdown("</div>", unsafe_allow_html=True)

with tab_geo:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("Geographic distribution")
    if "LAT" in df.columns and "LON" in df.columns:
        geo = df.copy()
        geo["LAT"] = pd.to_numeric(geo["LAT"], errors="coerce")
        geo["LON"] = pd.to_numeric(geo["LON"], errors="coerce")
        geo = geo.dropna(subset=["LAT", "LON"])

        hover_cols = [c for c in ["FIRST", "LAST", "CITY", "STATE", "AGE", "RACE", "ETHNICITY", "MONTH"] if c in geo.columns]
        color_col = "DECEASED" if "DECEASED" in geo.columns else None

        fig = px.scatter_mapbox(
            geo.head(5000),
            lat="LAT",
            lon="LON",
            color=color_col,
            hover_data=hover_cols,
            zoom=8,
            height=560,
            title="Patient locations (sampled if very large)",
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Map uses OpenStreetMap tiles (no token required).")
    else:
        st.info("LAT/LON columns not found.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.subheader("Data preview")
    st.dataframe(df.head(200), use_container_width=True, height=520)
    st.caption(f"Showing first 200 rows of the filtered dataset. Columns: {len(df.columns)}.")
    st.markdown("</div>", unsafe_allow_html=True)
