"""
Retail persona names and short stories for end-user UI (aligned with project README).
"""
from __future__ import annotations

import math

import pandas as pd

PERSONA_NAMES_5 = [
    "Prosperous Shoppers",
    "Web-Shrewd Shoppers",
    "Discount-Seeking Web Shoppers",
    "Web Enthusiasts with Frugal Habits",
    "Young Budget Shoppers",
]

# One-line + detail for dashboard cards (from README)
PERSONA_STORIES: dict[str, tuple[str, str]] = {
    "Prosperous Shoppers": (
        "Premium spenders & loyal buyers",
        "Higher income, strong total spend, often in-store and catalog. Usually open to campaigns and fewer children at home — ideal for a VIP loyalty tier.",
    ),
    "Web-Shrewd Shoppers": (
        "Active online researchers",
        "Solid income and purchase rhythm; mix of web and store; many web visits. Good candidates for digital perks and cross-channel offers.",
    ),
    "Discount-Seeking Web Shoppers": (
        "Price-sensitive web buyers",
        "Lower spend and frequency; likes deals and the website; often families. Win with clear promotions, not luxury positioning.",
    ),
    "Web Enthusiasts with Frugal Habits": (
        "Browsers who spend lightly",
        "Many site visits but modest baskets. Nudge with bundles, reminders, and easy first-purchase incentives.",
    ),
    "Young Budget Shoppers": (
        "Younger, budget-conscious",
        "Younger on average, lower income and small purchases; online often. Build long-term value with entry-level loyalty and education.",
    ),
}


def business_rank_column(cluster_summary: pd.DataFrame) -> str:
    """Pick a numeric column to rank segments (retail: monetary/income; else first numeric)."""
    for c in ("monetary", "income"):
        if c in cluster_summary.columns and pd.api.types.is_numeric_dtype(cluster_summary[c]):
            return c
    skip = {"cluster_id", "customer_count"}
    nums = [
        c
        for c in cluster_summary.columns
        if c not in skip and pd.api.types.is_numeric_dtype(cluster_summary[c])
    ]
    if nums:
        return sorted(nums)[0]
    return "cluster_id"


def persona_labels_for_clusters(cluster_summary: pd.DataFrame, k: int) -> dict[int, str]:
    if "cluster_id" not in cluster_summary.columns:
        return {}
    rank_col = business_rank_column(cluster_summary)
    if rank_col == "cluster_id":
        return {int(r["cluster_id"]): f"Customer group {int(r['cluster_id'])}" for _, r in cluster_summary.iterrows()}

    ranked = cluster_summary.sort_values(rank_col, ascending=False)["cluster_id"].tolist()
    use_readme_personas = k == 5 and "monetary" in cluster_summary.columns
    out: dict[int, str] = {}
    for rank, cid in enumerate(ranked):
        if use_readme_personas and rank < len(PERSONA_NAMES_5):
            out[int(cid)] = PERSONA_NAMES_5[rank]
        else:
            out[int(cid)] = f"Customer group {rank + 1}"
    return out


def apply_persona_column(df: pd.DataFrame, mapping: dict[int, str], cluster_col: str = "cluster_id") -> pd.Series:
    return df[cluster_col].map(lambda x: mapping.get(int(x), f"Group {x}"))


def story_for_persona(name: str) -> tuple[str, str]:
    return PERSONA_STORIES.get(name, ("Unique segment", "This group has its own shopping pattern — review spend, frequency, and channel mix to choose the right offer."))


_METRIC_LABELS: dict[str, str] = {
    "monetary": "total spend",
    "frequency": "purchase frequency",
    "income": "income",
    "avg_purchase_value": "basket value",
    "numwebvisitsmonth": "website visits",
    "numwebpurchases": "web purchases",
    "numstorepurchases": "store purchases",
    "numcatalogpurchases": "catalog purchases",
    "numdealspurchases": "deal usage",
    "children": "children at home",
    "age": "average age",
    "total_accepted_cmp": "campaign acceptance",
    "recency": "days since last purchase",
    "relationship_duration": "relationship length",
    "customer_count": "customers",
}

_METRIC_PRIORITY = [
    "monetary",
    "frequency",
    "avg_purchase_value",
    "income",
    "numwebpurchases",
    "numstorepurchases",
    "numcatalogpurchases",
    "numwebvisitsmonth",
    "numdealspurchases",
    "total_accepted_cmp",
    "children",
    "age",
    "recency",
    "relationship_duration",
]


def _band(summary: pd.DataFrame, column: str, value: float) -> str:
    series = summary[column].dropna()
    if series.empty or series.nunique() <= 1 or pd.isna(value):
        return "mid"
    low = float(series.quantile(0.33))
    high = float(series.quantile(0.67))
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "mid"


def _metric_label(column: str) -> str:
    return _METRIC_LABELS.get(column, column.replace("_", " "))


def _metric_phrase(column: str, direction: str) -> str:
    phrases = {
        "monetary": {"high": "higher total spend", "low": "lower total spend"},
        "frequency": {"high": "more frequent purchasing", "low": "less frequent purchasing"},
        "income": {"high": "higher income", "low": "lower income"},
        "avg_purchase_value": {"high": "larger basket values", "low": "smaller basket values"},
        "numwebvisitsmonth": {"high": "more website visits", "low": "fewer website visits"},
        "numwebpurchases": {"high": "more web purchases", "low": "fewer web purchases"},
        "numstorepurchases": {"high": "more in-store purchases", "low": "fewer in-store purchases"},
        "numcatalogpurchases": {"high": "more catalog purchases", "low": "fewer catalog purchases"},
        "numdealspurchases": {"high": "stronger deal usage", "low": "lighter deal usage"},
        "children": {"high": "larger households", "low": "smaller households"},
        "age": {"high": "an older customer profile", "low": "a younger customer profile"},
        "total_accepted_cmp": {"high": "more campaign acceptance", "low": "less campaign acceptance"},
        "recency": {"high": "less recent activity", "low": "more recent activity"},
        "relationship_duration": {"high": "longer customer relationships", "low": "shorter customer relationships"},
    }
    return phrases.get(column, {}).get(direction, f"{direction} {_metric_label(column)}")


def _join_phrases(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _format_value(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if math.isclose(value, round(value), abs_tol=1e-9):
        return f"{int(round(value)):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def dynamic_story_for_cluster(
    cluster_row: pd.Series,
    cluster_summary: pd.DataFrame,
    cluster_name: str,
    load_mode: str,
) -> tuple[str, str]:
    if load_mode != "marketing":
        numeric_cols = [
            c for c in cluster_summary.columns
            if c not in {"cluster_id", "customer_count"} and pd.api.types.is_numeric_dtype(cluster_summary[c])
        ]
        chosen = numeric_cols[:3]
        if not chosen:
            return ("Distinct pattern", "This group stands apart on the numeric columns available in the uploaded file.")
        highlights = []
        for col in chosen:
            state = _band(cluster_summary, col, cluster_row[col])
            if state == "mid":
                highlights.append(f"{_metric_label(col)} around {_format_value(cluster_row[col])}")
            else:
                highlights.append(f"{_metric_label(col)} {state} ({_format_value(cluster_row[col])})")
        return (
            "Generated from uploaded columns",
            f"{cluster_name} is defined from the numeric fields in your file, with notable values in {_join_phrases(highlights)}.",
        )

    high_cols: list[str] = []
    low_cols: list[str] = []
    for col in _METRIC_PRIORITY:
        if col not in cluster_summary.columns or not pd.api.types.is_numeric_dtype(cluster_summary[col]):
            continue
        state = _band(cluster_summary, col, cluster_row[col])
        if state == "high":
            high_cols.append(col)
        elif state == "low":
            low_cols.append(col)

    if "monetary" in high_cols and "frequency" in high_cols:
        tag = "High-value repeat buyers"
    elif "numwebvisitsmonth" in high_cols and "numwebpurchases" in high_cols:
        tag = "Digital-first shoppers"
    elif "numdealspurchases" in high_cols:
        tag = "Promotion-responsive buyers"
    elif "age" in low_cols and "monetary" in low_cols:
        tag = "Younger lower-spend shoppers"
    elif high_cols:
        tag = f"Stronger in {_metric_label(high_cols[0])}"
    elif low_cols:
        tag = f"Lower in {_metric_label(low_cols[0])}"
    else:
        tag = "Balanced customer profile"

    high_text = _join_phrases([_metric_phrase(col, "high") for col in high_cols[:3]])
    low_text = _join_phrases([_metric_phrase(col, "low") for col in low_cols[:2]])

    parts: list[str] = []
    if high_text:
        parts.append(f"Compared with the other groups, this segment shows {high_text}.")
    if low_text:
        parts.append(f"It stays lower on {low_text}.")

    customer_count = int(cluster_row["customer_count"]) if "customer_count" in cluster_row else None
    if customer_count is not None:
        parts.append(f"This group contains about {customer_count:,} customers in the current file.")

    if not parts:
        parts.append("This group has a distinct mix of value, channel usage, and customer profile compared with the rest of the dataset.")

    return tag, " ".join(parts)
