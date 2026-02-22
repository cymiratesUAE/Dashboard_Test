import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Patient Insights Dashboard",
    page_icon="🩺",
    layout="wide",
)

# -----------------------------
# Styling: white theme + blue/green accents + glowing tabs
# -----------------------------
ACCENT_BLUE = "#1E88E5"
ACCENT_GREEN = "#00C853"

CUSTOM_CSS = f"""
<style>
:root {{
  --accent-blue: {ACCENT_BLUE};
  --accent-green: {ACCENT_GREEN};
  --card: rgba(255,255,255,0.92);
  --shadow: 0 10px 25px rgba(0,0,0,0.06);
}}

/* App background */
.stApp {{
  background: radial-gradient(1200px 600px at 10% 5%, rgba(30,136,229,0.10), transparent 55%),
              radial-gradient(1000px 520px at 90% 0%, rgba(0,200,83,0.10), transparent 55%),
              #ffffff;
}}

/* Side bar */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(30,136,229,0.06), rgba(0,200,83,0.05));
  border-right: 1px solid rgba(0,0,0,0.06);
}}

/* Headings */
h1, h2, h3 {{
  letter-spacing: -0.02em;
}}

/* KPI cards */
.kpi-card {{
  background: var(--card);
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: var(--shadow);
  border-radius: 18px;
  padding: 16px 18px;
}}
.kpi-label {{
  font-size: 0.9rem;
  color: rgba(0,0,0,0.62);
}}
.kpi-value {{
  font-size: 1.8rem;
  font-weight: 700;
}}
.kpi-sub {{
  font-size: 0.85rem;
  color: rgba(0,0,0,0.55);
}}

/* Glowing tabs */
button[data-baseweb="tab"] {{
  background: rgba(255,255,255,0.8) !important;
  border-radius: 999px !important;
  margin-right: 10px !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
}}
button[data-baseweb="tab"] > div {{
  font-weight: 600 !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  border: 1px solid rgba(30,136,229,0.45) !important;
  box-shadow: 0 0 0 3px rgba(30,136,229,0.10), 0 0 18px rgba(0,200,83,0.18) !important;
  background: linear-gradient(90deg, rgba(30,136,229,0.12), rgba(0,200,83,0.10)) !important;
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{
  background: var(--card);
  border-radius: 18px;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: var(--shadow);
  padding: 8px;
}}

/* Buttons */
.stButton>button, .stDownloadButton>button {{
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.10);
  box-shadow: 0 8px 18px rgba(0,0,0,0.05);
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
  border-color: rgba(30,136,229,0.45);
  box-shadow: 0 0 0 3px rgba(30,136,229,0.10), 0 0 18px rgba(0,200,83,0.16);
}}

/* Hide Streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------

def _safe_to_datetime(s: pd.Series) -> pd.Series:
    """Parse a date column, returning NaT on failures."""
    return pd.to_datetime(s, errors="coerce", utc=False)


def _add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Standardize common columns if present
    for col in ["BIRTHDATE", "DEATHDATE"]:
        if col in out.columns:
            out[col] = _safe_to_datetime(out[col])

    # Alive/Deceased flags
    if "DEATHDATE" in out.columns:
        out["IS_DECEASED"] = out["DEATHDATE"].notna()
    else:
        out["IS_DECEASED"] = False

    # Age (in years) at today or death
    if "BIRTHDATE" in out.columns:
        ref = out["DEATHDATE"].where(out["IS_DECEASED"], pd.Timestamp.today())
        out["AGE_YEARS"] = ((ref - out["BIRTHDATE"]).dt.days / 365.25).round(1)
        out.loc[out["AGE_YEARS"].lt(0), "AGE_YEARS"] = np.nan
        out["BIRTH_YEAR"] = out["BIRTHDATE"].dt.year
    else:
        out["AGE_YEARS"] = np.nan
        out["BIRTH_YEAR"] = np.nan

    # A gentle normalization for categorical filters
    for c in ["GENDER", "RACE", "ETHNICITY", "MARITAL", "CITY", "STATE", "COUNTY"]:
        if c in out.columns:
            out[c] = out[c].astype(str).replace({"nan": np.nan}).str.strip()

    return out


@st.cache_data(show_spinner=False)
def load_base_data() -> pd.DataFrame:
    return pd.read_csv("data/patients.csv")


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    # Try utf-8-sig to handle BOM
    content = uploaded_file.read()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc)
        except Exception:
            continue
    # Last resort
    return pd.read_csv(io.BytesIO(content))


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def pretty_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"


# -----------------------------
# Data state
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = load_base_data()

if "last_upload_month" not in st.session_state:
    st.session_state.last_upload_month = ""

# -----------------------------
# Sidebar: upload + filters
# -----------------------------
with st.sidebar:
    st.markdown("## 📦 Data")

    uploaded = st.file_uploader(
        "Upload a new month CSV (same columns recommended)",
        type=["csv"],
        help="Tip: Keep column names consistent. We'll merge/append and keep your original rows.",
    )

    month_label = st.text_input(
        "Month label for upload (optional)",
        value=st.session_state.last_upload_month or datetime.today().strftime("%Y-%m"),
        help="Example: 2026-02",
    )

    col_add_month = st.checkbox("Add/overwrite MONTH column from label", value=True)

    if uploaded is not None:
        try:
            new_df = read_uploaded_csv(uploaded)
            if col_add_month:
                new_df["MONTH"] = month_label

            # Combine
            combined = pd.concat([st.session_state.data, new_df], ignore_index=True)

            # De-duplicate using Id if available
            if "Id" in combined.columns:
                combined = combined.drop_duplicates(subset=["Id"], keep="last")

            st.session_state.data = combined
            st.session_state.last_upload_month = month_label
            st.success(f"Loaded upload and updated dataset. Rows now: {len(combined):,}")
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")

    if st.button("Reset to original data", use_container_width=True):
        st.session_state.data = load_base_data()
        st.toast("Reset complete ✅")

    st.divider()
    st.markdown("## 🎛️ Filters")

# Prepare derived fields for filtering/analysis
base = _add_derived_fields(st.session_state.data)

# Build filters (in sidebar)
with st.sidebar:
    # Helper to build multiselects safely
    def multiselect_if_exists(label, col, default_all=True):
        if col not in base.columns:
            return None
        vals = sorted([v for v in base[col].dropna().unique().tolist() if str(v).strip() != ""])
        if not vals:
            return None
        default = vals if default_all else []
        return st.multiselect(label, vals, default=default)

    gender_sel = multiselect_if_exists("Gender", "GENDER")
    race_sel = multiselect_if_exists("Race", "RACE")
    eth_sel = multiselect_if_exists("Ethnicity", "ETHNICITY")
    marital_sel = multiselect_if_exists("Marital", "MARITAL")
    state_sel = multiselect_if_exists("State", "STATE")

    city_sel = None
    if "CITY" in base.columns:
        # City list can be long: use a searchable multiselect with smaller default
        city_vals = sorted([v for v in base["CITY"].dropna().unique().tolist() if str(v).strip() != ""])
        city_sel = st.multiselect("City (optional)", city_vals, default=[])

    alive_mode = st.radio(
        "Life status",
        ["All", "Alive", "Deceased"],
        horizontal=True,
        index=0,
    )

    # Birth year range
    if "BIRTH_YEAR" in base.columns and base["BIRTH_YEAR"].notna().any():
        y_min = int(np.nanmin(base["BIRTH_YEAR"]))
        y_max = int(np.nanmax(base["BIRTH_YEAR"]))
        birth_year_range = st.slider("Birth year range", y_min, y_max, (y_min, y_max))
    else:
        birth_year_range = None

    # Age range
    if "AGE_YEARS" in base.columns and base["AGE_YEARS"].notna().any():
        a_min = float(np.nanmin(base["AGE_YEARS"]))
        a_max = float(np.nanmax(base["AGE_YEARS"]))
        age_range = st.slider("Age (years)", float(np.floor(a_min)), float(np.ceil(a_max)), (float(np.floor(a_min)), float(np.ceil(a_max))))
    else:
        age_range = None

    st.divider()
    st.markdown("## ⬇️ Downloads")

# Apply filters
filtered = base.copy()

def _apply_multisel(df, col, sel):
    if sel is None:
        return df
    return df[df[col].isin(sel)]

filtered = _apply_multisel(filtered, "GENDER", gender_sel)
filtered = _apply_multisel(filtered, "RACE", race_sel)
filtered = _apply_multisel(filtered, "ETHNICITY", eth_sel)
filtered = _apply_multisel(filtered, "MARITAL", marital_sel)
filtered = _apply_multisel(filtered, "STATE", state_sel)

if city_sel is not None and len(city_sel) > 0:
    filtered = filtered[filtered["CITY"].isin(city_sel)]

if alive_mode == "Alive":
    filtered = filtered[~filtered["IS_DECEASED"]]
elif alive_mode == "Deceased":
    filtered = filtered[filtered["IS_DECEASED"]]

if birth_year_range is not None and "BIRTH_YEAR" in filtered.columns:
    filtered = filtered[filtered["BIRTH_YEAR"].between(birth_year_range[0], birth_year_range[1])]

if age_range is not None and "AGE_YEARS" in filtered.columns:
    filtered = filtered[filtered["AGE_YEARS"].between(age_range[0], age_range[1])]

# Downloads in sidebar
with st.sidebar:
    st.download_button(
        "Download filtered CSV",
        data=df_to_csv_bytes(filtered),
        file_name="patients_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Download full combined dataset",
        data=df_to_csv_bytes(base.drop(columns=[c for c in ["AGE_YEARS", "BIRTH_YEAR", "IS_DECEASED"] if c in base.columns], errors="ignore")),
        file_name="patients_combined.csv",
        mime="text/csv",
        use_container_width=True,
        help="This includes any uploads you've appended during this session.",
    )

# -----------------------------
# Header
# -----------------------------
st.title("🩺 Patient Insights Dashboard")
st.caption("Interactive exploration of patient demographics and outcomes. White theme, blue/green accents, and upload-to-update built in.")

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

n_total = len(filtered)

n_deceased = int(filtered["IS_DECEASED"].sum()) if "IS_DECEASED" in filtered.columns else 0
n_alive = n_total - n_deceased

median_age = float(np.nanmedian(filtered["AGE_YEARS"])) if "AGE_YEARS" in filtered.columns and filtered["AGE_YEARS"].notna().any() else np.nan
pct_deceased = (n_deceased / n_total * 100) if n_total > 0 else 0.0

# Approximate "data completeness" score: fraction of non-null across key columns
key_cols = [c for c in ["BIRTHDATE", "GENDER", "RACE", "ETHNICITY", "CITY", "STATE"] if c in filtered.columns]
if n_total > 0 and key_cols:
    completeness = float(filtered[key_cols].notna().mean(axis=1).mean() * 100)
else:
    completeness = np.nan

kpis = [
    ("Patients (filtered)", pretty_int(n_total), "Rows after applying your filters"),
    ("Alive", pretty_int(n_alive), "Based on missing/present DEATHDATE"),
    ("Deceased %", f"{pct_deceased:.1f}%", "In the filtered population"),
    ("Median age", f"{median_age:.1f}" if np.isfinite(median_age) else "-", "Age today or at death"),
]

for c, (lab, val, sub) in zip([col1, col2, col3, col4], kpis):
    with c:
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div class='kpi-label'>{lab}</div>
              <div class='kpi-value'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Optional extra stat below KPIs
if np.isfinite(completeness):
    st.markdown(
        f"<div style='margin-top:10px; color: rgba(0,0,0,0.60)'>📌 Data completeness (key columns): <b>{completeness:.1f}%</b></div>",
        unsafe_allow_html=True,
    )

# -----------------------------
# Tabs
# -----------------------------

tab_overview, tab_geo, tab_outcomes, tab_table, tab_about = st.tabs(
    ["Overview 📊", "Geo 🌍", "Outcomes 💚", "Data Table 🧾", "About 🧠"]
)

# ============ Overview ============
with tab_overview:
    left, right = st.columns([1.2, 1])

    # Age distribution
    with left:
        st.subheader("Age distribution")
        if "AGE_YEARS" in filtered.columns and filtered["AGE_YEARS"].notna().any():
            fig_age = px.histogram(
                filtered,
                x="AGE_YEARS",
                nbins=35,
                title=None,
                opacity=0.95,
            )
            fig_age.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
            )
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.info("No BIRTHDATE column (or couldn't parse dates), so age visuals are hidden.")

    # Categorical breakdown
    with right:
        st.subheader("Composition")
        cat = st.selectbox(
            "Break down by",
            options=[c for c in ["GENDER", "RACE", "ETHNICITY", "MARITAL", "STATE", "CITY"] if c in filtered.columns],
            index=0 if any(c in filtered.columns for c in ["GENDER", "RACE", "ETHNICITY", "MARITAL", "STATE", "CITY"]) else None,
        )

        if cat and cat in filtered.columns:
            counts = filtered[cat].fillna("(missing)").value_counts().head(12)
            fig_cat = px.bar(
                x=counts.values,
                y=counts.index,
                orientation="h",
                labels={"x": "Patients", "y": cat},
            )
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Pick a column to see a breakdown.")

    st.divider()

    # Fancy analysis: risk signals / cohort view
    st.subheader("Fancy analysis: mortality signal by cohort")

    if "AGE_YEARS" in filtered.columns and filtered["AGE_YEARS"].notna().any():
        cohort_col = st.selectbox(
            "Cohort dimension",
            options=[c for c in ["GENDER", "RACE", "ETHNICITY", "MARITAL", "STATE"] if c in filtered.columns],
            index=0,
        )

        # Create age bands
        age_bins = [0, 18, 30, 40, 50, 60, 70, 80, 120]
        labels = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        tmp = filtered.copy()
        tmp["AGE_BAND"] = pd.cut(tmp["AGE_YEARS"], bins=age_bins, labels=labels, right=False)

        grp = (
            tmp.groupby([cohort_col, "AGE_BAND"], dropna=False)
            .agg(patients=("Id" if "Id" in tmp.columns else tmp.columns[0], "count"), deceased=("IS_DECEASED", "sum"))
            .reset_index()
        )
        grp["deceased_rate"] = np.where(grp["patients"] > 0, grp["deceased"] / grp["patients"], np.nan)

        # Heatmap
        pivot = grp.pivot(index=cohort_col, columns="AGE_BAND", values="deceased_rate").fillna(0)
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(x) for x in pivot.columns.tolist()],
                y=[str(y) for y in pivot.index.tolist()],
                colorbar=dict(title="Deceased rate"),
            )
        )
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.caption("This is a descriptive signal, not a clinical prediction. Use it to spot patterns worth investigating.")
    else:
        st.info("Fancy cohort analysis needs BIRTHDATE and (optionally) DEATHDATE.")

# ============ Geo ============
with tab_geo:
    st.subheader("Geographic view")
    if all(c in filtered.columns for c in ["LAT", "LON"]):
        geo_df = filtered.dropna(subset=["LAT", "LON"]).copy()
        if len(geo_df) == 0:
            st.info("No rows with LAT/LON after filtering.")
        else:
            # For performance: sample if huge
            max_points = 5000
            if len(geo_df) > max_points:
                geo_df = geo_df.sample(max_points, random_state=7)
                st.caption(f"Showing a random sample of {max_points:,} points for performance.")

            hover_cols = [c for c in ["FIRST", "LAST", "CITY", "STATE", "RACE", "ETHNICITY", "AGE_YEARS"] if c in geo_df.columns]
            fig_map = px.scatter_mapbox(
                geo_df,
                lat="LAT",
                lon="LON",
                hover_data=hover_cols,
                zoom=8,
                height=560,
            )
            # Use open-street-map style (no token needed)
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.caption("Tip: zoom and pan to explore clusters. Combine with filters for focused analysis.")
    else:
        st.info("LAT/LON columns not found. If your future dataset includes coordinates, this tab will light up.")

# ============ Outcomes ============
with tab_outcomes:
    st.subheader("Outcomes")

    cols = st.columns([1, 1])

    # Alive vs deceased donut
    with cols[0]:
        if "IS_DECEASED" in filtered.columns:
            donut = pd.DataFrame(
                {
                    "status": ["Alive", "Deceased"],
                    "count": [n_alive, n_deceased],
                }
            )
            fig_donut = px.pie(donut, names="status", values="count", hole=0.55)
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No DEATHDATE column found.")

    # Deaths by year
    with cols[1]:
        if "DEATHDATE" in filtered.columns and filtered["DEATHDATE"].notna().any():
            tmp = filtered.copy()
            tmp["DEATH_YEAR"] = tmp["DEATHDATE"].dt.year
            deaths_by_year = tmp[tmp["IS_DECEASED"]].groupby("DEATH_YEAR").size().reset_index(name="deaths")
            fig_deaths = px.line(deaths_by_year, x="DEATH_YEAR", y="deaths", markers=True)
            fig_deaths.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
            )
            st.plotly_chart(fig_deaths, use_container_width=True)
        else:
            st.info("Deaths-by-year chart needs DEATHDATE.")

    st.divider()

    st.subheader("Segment comparison")
    # Compare two groups on deceased rate
    opts = [c for c in ["GENDER", "RACE", "ETHNICITY", "MARITAL", "STATE", "CITY"] if c in filtered.columns]
    if opts and "IS_DECEASED" in filtered.columns:
        seg = st.selectbox("Compare mortality across", opts, index=0)
        tmp = filtered.copy()
        comp = (
            tmp.groupby(seg, dropna=False)
            .agg(patients=("Id" if "Id" in tmp.columns else tmp.columns[0], "count"), deceased=("IS_DECEASED", "sum"))
            .reset_index()
        )
        comp["deceased_rate"] = np.where(comp["patients"] > 0, comp["deceased"] / comp["patients"], np.nan)
        comp = comp.sort_values(["patients"], ascending=False).head(15)

        fig_seg = px.bar(
            comp,
            x="deceased_rate",
            y=seg,
            orientation="h",
            hover_data=["patients", "deceased"],
            labels={"deceased_rate": "Deceased rate"},
        )
        fig_seg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=460,
        )
        st.plotly_chart(fig_seg, use_container_width=True)

        st.caption("Interpretation hint: high rates with tiny sample sizes can be noisy. Hover to see counts.")
    else:
        st.info("Segment comparison needs at least one categorical column and DEATHDATE/IS_DECEASED.")

# ============ Data Table ============
with tab_table:
    st.subheader("Filtered data")
    show_cols = st.multiselect(
        "Choose columns",
        options=filtered.columns.tolist(),
        default=[c for c in ["Id", "FIRST", "LAST", "GENDER", "AGE_YEARS", "CITY", "STATE", "RACE", "ETHNICITY", "IS_DECEASED"] if c in filtered.columns],
    )
    st.dataframe(filtered[show_cols], use_container_width=True, height=520)

# ============ About ============
with tab_about:
    st.subheader("What this dashboard does")
    st.markdown(
        """
        - **Filters** in the sidebar let you slice the population by demographics and geography.
        - **Uploads** allow you to append a new month of data. If you enable it, the app adds a `MONTH` column from your label.
        - **Fancy analysis** includes a cohort heatmap (mortality rate by age band and segment) and segment comparisons.

        **Publishing tip**: Deploy this with Streamlit Community Cloud, Streamlit for Teams, Azure App Service, or any container platform.
        """
    )

    st.subheader("Expected columns")
    st.markdown(
        """
        The app auto-detects columns. For the best experience, keep these (if available):
        - `BIRTHDATE` (YYYY-MM-DD)
        - `DEATHDATE` (YYYY-MM-DD or blank)
        - `GENDER`, `RACE`, `ETHNICITY`, `MARITAL`, `CITY`, `STATE`, `COUNTY`
        - `LAT`, `LON`

        If your next-month file has extra columns, the app keeps them too.
        """
    )
