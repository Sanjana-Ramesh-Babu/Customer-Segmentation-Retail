# Power BI dashboard (Retail K-Means segmentation)

This folder holds **instructions** and generated **CSV exports** for building a Power BI report aligned with the Streamlit app and `notebooks/modelling.ipynb`.

## 1. Generate data

From the project root (after installing dependencies):

```bash
python -m scripts.export_powerbi_datasets
```

Or use the Streamlit app: **Power BI export** tab → **Export CSVs for Power BI**.

Files are written to `powerbi/exports/`:

| File | Purpose |
|------|---------|
| `customers_segmented.csv` | One row per customer: `cluster_id`, PCA coordinates, key attributes |
| `cluster_summary.csv` | Per-cluster means and `customer_count` |
| `segmentation_metrics.csv` | Silhouette, Davies–Bouldin, Calinski–Harabasz, inertia, Between/Total SS |
| `silhouette_by_k.csv` | Silhouette and inertia for K = 2…10 (tuning chart) |
| `pca_components.csv` | PCA loadings (interpretability) |

## 2. Build the Power BI model

1. Open **Power BI Desktop**.
2. **Get data** → **Text/CSV** → select `customers_segmented.csv` and `cluster_summary.csv` (or **Folder** → `powerbi/exports` and choose files).
3. In **Model view**, relate `customers_segmented[cluster_id]` to `cluster_summary[cluster_id]` (*many-to-one*).
4. Set `cluster_id` data type to **Whole number** on both tables.

## 3. Suggested visuals

- **Cluster sizes:** bar chart — `customer_count` from `cluster_summary` by `cluster_id`.
- **Segment separation:** scatter — `pca_1` vs `pca_2`, legend = `cluster_id` (from `customers_segmented`).
- **RFM-style profile:** clustered column chart — average `monetary`, `frequency`, `recency` by `cluster_id` (use `cluster_summary`).
- **KPI card:** import `segmentation_metrics` and show **Silhouette** or **Between / Total SS** as single values.
- **K tuning:** line chart from `silhouette_by_k` — `k` on axis, `silhouette_score` and/or `inertia`.

## 4. Refresh

Re-run the export script or Streamlit export after changing data or K; in Power BI use **Refresh** (same folder paths).

Note: `.pbix` files are binary and are not committed here; keep your report locally or in shared storage.
