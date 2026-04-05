"""
Retail persona names and short stories for end-user UI (aligned with project README).
"""
from __future__ import annotations

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
