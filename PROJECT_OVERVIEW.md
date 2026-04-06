# Project Overview

## What This Project Is

This is a **customer segmentation** project for a retail business.

The core idea is:

1. Take a customer-level dataset.
2. Engineer retail behavior features such as spend, purchase frequency, children at home, campaign acceptance, and web/store/catalog activity.
3. Preprocess the data.
4. Reduce dimensions with PCA.
5. Cluster customers with **K-Means**.
6. Present the results in:
   - a **Streamlit app**
   - exported **CSV files for Power BI**
   - notebooks and report images for analysis/storytelling

The business framing in the repo is: identify shopper segments, profile them, and design a loyalty program around the best segment ("Prosperous Shoppers").

## What Runs Today

There are three practical entry points:

### 1. Streamlit app

File: `app/streamlit_app.py`

Run:

```bash
streamlit run app/streamlit_app.py
```

What it does:

- Loads the sample dataset by default.
- Lets a user upload a CSV/TXT customer file.
- Lets a user choose the number of clusters.
- Runs the segmentation pipeline.
- Shows cluster quality and customer-group outputs.
- Writes Power BI export CSVs into `powerbi/exports/`.

### 2. Segmentation pipeline

File: `scripts/segmentation_pipeline.py`

This is the main reusable backend pipeline. It:

- loads data
- decides whether the file matches the expected marketing-campaign schema or should be treated as a generic numeric table
- preprocesses features
- runs PCA
- fits K-Means
- computes clustering metrics
- returns dataframes for downstream use

### 3. Power BI export script

File: `scripts/export_powerbi_datasets.py`

Run:

```bash
python -m scripts.export_powerbi_datasets
```

This generates:

- `powerbi/exports/customers_segmented.csv`
- `powerbi/exports/cluster_summary.csv`
- `powerbi/exports/segmentation_metrics.csv`
- `powerbi/exports/silhouette_by_k.csv`
- `powerbi/exports/pca_components.csv`

## Real Data Flow

The actual code flow is:

1. Input file is loaded through `scripts/table_io.py`.
2. If the file matches the marketing dataset schema, it uses `feature_engineering()` from `scripts/modelling_utils.py`.
3. Otherwise, it falls back to clustering on a cleaned numeric-only table.
4. Preprocessing is built in `scripts/segmentation_pipeline.py`:
   - numeric columns: `KNNImputer` + `StandardScaler`
   - categorical columns: `OrdinalEncoder` + `StandardScaler`
5. PCA is applied, usually with 3 components.
6. K-Means is trained.
7. Metrics are computed in `scripts/cluster_metrics.py`.
8. Persona labels are assigned in `app/personas.py`.
9. Results are shown in Streamlit and exported for Power BI.

## Important Files

### App / UI

- `app/streamlit_app.py`: main dashboard
- `app/personas.py`: maps clusters to friendly shopper/persona names and descriptions

### Core pipeline

- `scripts/segmentation_pipeline.py`: main orchestration
- `scripts/table_io.py`: file loading, delimiter/encoding detection, schema matching, generic fallback
- `scripts/modelling_utils.py`: feature engineering and older modeling helpers
- `scripts/cluster_metrics.py`: silhouette, Davies-Bouldin, Calinski-Harabasz, variance ratio, centroid distance
- `scripts/export_powerbi_datasets.py`: CLI export entry point

### Data / artifacts

- `artifacts/marketing_campaign.csv`: sample raw dataset copy
- `artifacts/model.pkl`: saved trained model
- `artifacts/preprocessor.pkl`: saved preprocessing artifact
- `artifacts/prepared_data.pkl`: prepared dataset artifact
- `uploads/uploaded_customers.csv`: latest uploaded file from the app

### Analysis / storytelling

- `notebooks/eda.ipynb`: exploratory analysis
- `notebooks/modelling.ipynb`: modeling workflow
- `reports/*.png`: charts used in README/storytelling
- `README.md`: business narrative and project explanation
- `powerbi/README.md`: Power BI instructions

## Expected Dataset Shape

The original project expects a **customer-level marketing table**, not transaction rows.

Examples of expected fields:

- customer ID
- birth year
- education
- marital status
- income
- children at home
- date customer joined
- recency
- spend by category
- purchase counts by channel
- campaign acceptance columns

The loader in `scripts/table_io.py` is more flexible than the README suggests:

- it auto-detects delimiter and encoding
- it normalizes header names using aliases
- if the full marketing schema is missing, it can still cluster a generic tabular file using numeric columns only

## Key Engineered Features

When the marketing schema is detected, `feature_engineering()` creates or derives:

- `total_accepted_cmp`
- `children`
- `age`
- `relationship_duration`
- `frequency`
- `monetary`
- `avg_purchase_value`

It also removes some outliers and drops columns considered irrelevant for clustering.

## Current Segment Naming Logic

If there are exactly 5 clusters and the summary includes `monetary`, the app assigns these names:

1. Prosperous Shoppers
2. Web-Shrewd Shoppers
3. Discount-Seeking Web Shoppers
4. Web Enthusiasts with Frugal Habits
5. Young Budget Shoppers

If not, it falls back to generic labels like `Customer group 1`.

## What Looks Old vs New

This repo has two layers:

- **Older analysis layer**: notebooks, README story, legacy helper functions in `scripts/modelling_utils.py`
- **Newer productized layer**: Streamlit app, robust file loader, reusable pipeline, Power BI export flow

So if you need to change behavior, the files that matter most are usually:

- `app/streamlit_app.py`
- `scripts/segmentation_pipeline.py`
- `scripts/table_io.py`
- `app/personas.py`

The notebooks are more useful for reference than for runtime behavior.

## Things To Watch Before Making Changes

### 1. Feature engineering uses a hard-coded year for age

In `scripts/modelling_utils.py`, age is computed as:

```python
feat_eng_df['age'] = 2023 - feat_eng_df['year_birth']
```

That means age becomes stale over time.

### 2. Relationship duration uses the current system year

This makes results time-dependent:

```python
current_date = datetime.today()
feat_eng_df['relationship_duration'] = (current_date.year - feat_eng_df['dt_customer'].dt.year)
```

That may be intended, but it means reruns in different years can change clustering features.

### 3. Saved artifacts may not match current code exactly

There are `.pkl` artifacts in `artifacts/`, but the active app path recomputes segmentation directly through the pipeline. If you change preprocessing or features, treat old artifacts carefully.

### 4. README is partly narrative, not always the source of truth

The README explains the business story well, but the code now supports more flexible input handling than the original project description suggests.

## If You Need To Modify This Project

Use this quick guide:

- Change app behavior/UI: `app/streamlit_app.py`
- Change cluster naming or persona copy: `app/personas.py`
- Change preprocessing, PCA, K-Means, metrics, or output tables: `scripts/segmentation_pipeline.py`
- Change accepted input columns / file parsing: `scripts/table_io.py`
- Change marketing feature engineering logic: `scripts/modelling_utils.py`
- Change Power BI output files: `scripts/export_powerbi_datasets.py`

## Recommended First Commands

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
python -m scripts.export_powerbi_datasets
```

## Bottom Line

This is a **retail customer clustering application** built around a K-Means segmentation pipeline. The notebooks and README explain the original data science project, while the current runnable product is the Streamlit app plus the reusable scripts under `scripts/`.
