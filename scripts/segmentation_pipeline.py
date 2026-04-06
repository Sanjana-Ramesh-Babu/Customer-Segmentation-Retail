"""
End-to-end K-Means segmentation pipeline aligned with notebooks/modelling.ipynb.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from scripts.cluster_metrics import cluster_metrics_bundle, metrics_summary_rows
from scripts.modelling_utils import feature_engineering
from scripts.table_io import load_for_segmentation


@dataclass
class SegmentationResult:
    clustering_df: pd.DataFrame
    customer_ids: pd.Series
    load_mode: str
    X_pca: np.ndarray
    pca: PCA
    kmeans: KMeans
    labels: np.ndarray
    metrics: dict
    metrics_table: pd.DataFrame
    customer_segment: pd.DataFrame
    cluster_summary: pd.DataFrame
    silhouette_by_k: pd.DataFrame


def _build_preprocessor(clustering_df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    numerical_features = clustering_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = clustering_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_pipeline = Pipeline(
        steps=[
            ("knn_imputer", KNNImputer()),
            ("std_scaler", StandardScaler()),
        ]
    )
    transformers: list = [("num", num_pipeline, numerical_features)]
    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("ordinal_encoder", OrdinalEncoder()),
                ("std_scaler", StandardScaler()),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numerical_features, categorical_features


def _silhouette_sweep_k(
    X: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> pd.DataFrame:
    from sklearn.metrics import silhouette_score

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=100,
            max_iter=300,
            random_state=random_state,
        )
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, metric="euclidean")
        rows.append({"k": k, "silhouette_score": sil, "inertia": km.inertia_})
    return pd.DataFrame(rows)


def run_segmentation(
    data_path: Optional[Path] = None,
    raw_bytes: Optional[bytes] = None,
    n_clusters: int = 5,
    n_components: int = 3,
    random_state: int = 42,
) -> SegmentationResult:
    """
    Load data (any common CSV delimiter + header aliases), then feature engineering, PCA, K-Means.
    Provide either `data_path` or `raw_bytes` (e.g. an uploaded file).
    """
    if raw_bytes is not None:
        loaded, ids_hint, load_mode = load_for_segmentation(raw=raw_bytes)
    else:
        if data_path is None:
            root = Path(__file__).resolve().parents[1]
            data_path = root / "notebooks" / "data" / "marketing_campaign.csv"
        loaded, ids_hint, load_mode = load_for_segmentation(path=Path(data_path))

    if load_mode == "marketing":
        clustering_df, customer_ids = feature_engineering(loaded)
    else:
        clustering_df, customer_ids = loaded, ids_hint

    preprocessor, _, _ = _build_preprocessor(clustering_df)
    prepared = preprocessor.fit_transform(clustering_df)
    prepared_df = pd.DataFrame(prepared, columns=clustering_df.columns)

    n_comp_eff = int(
        min(
            n_components,
            max(1, clustering_df.shape[1]),
            max(1, clustering_df.shape[0]),
        )
    )
    pca = PCA(n_components=n_comp_eff, random_state=random_state)
    X_pca = pca.fit_transform(prepared_df)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=100,
        max_iter=300,
        random_state=random_state,
    )
    kmeans.fit(X_pca)
    labels = kmeans.labels_

    metrics = cluster_metrics_bundle(
        X=X_pca,
        labels=labels,
        inertia=kmeans.inertia_,
        centroids=kmeans.cluster_centers_,
    )
    metrics_table = metrics_summary_rows(metrics)

    customer_segment = clustering_df.copy()
    customer_segment.insert(0, "customer_id", customer_ids.values)
    customer_segment["cluster_id"] = labels
    for i in range(X_pca.shape[1]):
        customer_segment[f"pca_{i + 1}"] = X_pca[:, i]

    num_cols = clustering_df.select_dtypes("number").columns.to_list()
    counts = (
        customer_segment.groupby("cluster_id")["customer_id"]
        .count()
        .reset_index(name="customer_count")
    )
    cluster_summary = customer_segment.groupby("cluster_id", as_index=False)[num_cols].mean()
    cluster_summary = cluster_summary.merge(counts, on="cluster_id")

    silhouette_by_k = _silhouette_sweep_k(X_pca, random_state=random_state)

    return SegmentationResult(
        clustering_df=clustering_df,
        customer_ids=customer_ids,
        load_mode=load_mode,
        X_pca=X_pca,
        pca=pca,
        kmeans=kmeans,
        labels=labels,
        metrics=metrics,
        metrics_table=metrics_table,
        customer_segment=customer_segment,
        cluster_summary=cluster_summary,
        silhouette_by_k=silhouette_by_k,
    )


def export_powerbi_csvs(result: SegmentationResult, out_dir: Path) -> dict[str, Path]:
    """Write CSV files for Power BI (Get Data → Text/CSV or folder)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    p1 = out_dir / "customers_segmented.csv"
    p2 = out_dir / "cluster_summary.csv"
    p3 = out_dir / "segmentation_metrics.csv"
    p4 = out_dir / "silhouette_by_k.csv"
    p5 = out_dir / "pca_components.csv"

    result.customer_segment.to_csv(p1, index=False)
    result.cluster_summary.to_csv(p2, index=False)
    result.metrics_table.to_csv(p3, index=False)
    result.silhouette_by_k.to_csv(p4, index=False)

    comp = pd.DataFrame(
        result.pca.components_,
        columns=result.clustering_df.columns,
        index=[f"PC{i+1}" for i in range(result.pca.n_components_)],
    )
    comp.to_csv(p5)

    paths["customers_segmented"] = p1
    paths["cluster_summary"] = p2
    paths["segmentation_metrics"] = p3
    paths["silhouette_by_k"] = p4
    paths["pca_components"] = p5
    return paths
