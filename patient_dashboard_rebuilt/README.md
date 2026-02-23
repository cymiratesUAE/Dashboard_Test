# Patient Insights Dashboard (Streamlit)

Interactive dashboard with filters, tabs, segment analysis, cohort heatmap, map, and an uploader to add new monthly datasets.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Cloud, create a new app:
   - Repository: your repo
   - Branch: main
   - Main file path: `app.py`
3. Deploy.

Notes:
- Default dataset lives at `data/patients.csv`.
- You can upload a new CSV from the sidebar. If you provide a month label, it will be stored in a `MONTH` column.
- Use the download buttons to export the filtered view or the combined dataset.

## Data expectations

Default columns expected (from Synthea-style patient export):
`Id, BIRTHDATE, DEATHDATE, FIRST, LAST, GENDER, RACE, ETHNICITY, MARITAL, CITY, STATE, COUNTY, ZIP, LAT, LON`

The app will still run if some optional columns are missing; it will adapt charts accordingly.
