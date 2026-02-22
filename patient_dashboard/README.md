# Patient Insights Dashboard (Streamlit)

## Run locally

```bash
cd patient_dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Update data each month

- Use the sidebar **Upload a new month CSV**.
- Optionally set a **Month label** (e.g., `2026-02`) and tick **Add/overwrite MONTH**.
- Download the **full combined dataset** to keep as your new master file.

## Publish

Common options:
- Streamlit Community Cloud
- Streamlit for Teams
- Azure App Service / AWS / GCP (container)

Point the deployment to `app.py` and include the `requirements.txt`.

## Files

- `app.py` Streamlit dashboard
- `data/patients.csv` starter dataset
- `requirements.txt` dependencies
