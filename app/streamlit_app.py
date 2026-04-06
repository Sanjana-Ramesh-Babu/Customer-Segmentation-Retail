"""
Retail Store Customer Segmentation — end-user dashboard (plain language).

Run from project root:
  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_APP = Path(__file__).resolve().parent
for _p in (_ROOT, _APP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from personas import (
    apply_persona_column,
    business_rank_column,
    dynamic_story_for_cluster,
    persona_labels_for_clusters,
)
from scripts.exception import CustomException
from scripts.segmentation_pipeline import export_powerbi_csvs as write_powerbi_csvs
from scripts.segmentation_pipeline import run_segmentation

st.set_page_config(
    page_title="Customer Segmentation | Retail",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UPLOAD_DIR = _ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
  html, body, [class*="css"]  { font-family: 'DM Sans', 'Segoe UI', system-ui, sans-serif; }
  .block-container { padding-top: 1rem !important; max-width: 1180px; }
  .top-banner {
    background: linear-gradient(145deg, #1b4332 0%, #2d6a4f 45%, #40916c 100%);
    color: #fff;
    padding: 2rem 1.75rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 40px rgba(27, 67, 50, 0.25);
  }
  .top-banner h1 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.03em; }
  .top-banner .sub { margin: 0.65rem 0 0 0; font-size: 1.05rem; opacity: 0.95; line-height: 1.5; max-width: 42rem; }
  .top-banner .fine { margin: 0.85rem 0 0 0; font-size: 0.82rem; opacity: 0.85; }
  .input-panel {
    background: #ffffff;
    border: 1px solid #e8e6e1;
    border-radius: 14px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
  }
  .kpi-row { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; }
  .kpi {
    flex: 1; min-width: 140px;
    background: #fff; border: 1px solid #e8e6e1; border-radius: 12px;
    padding: 1rem 1.1rem;
  }
  .kpi .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; }
  .kpi .value { font-size: 1.45rem; font-weight: 700; color: #1b4332; margin-top: 0.2rem; }
  .kpi .hint { font-size: 0.8rem; color: #6b7280; margin-top: 0.35rem; line-height: 1.35; }
  .card-persona {
    background: #fff;
    border: 1px solid #e8e6e1;
    border-radius: 14px;
    padding: 1.1rem 1.15rem;
    height: 100%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
  }
  .card-persona h4 { margin: 0 0 0.35rem 0; font-size: 1rem; color: #1b4332; }
  .card-persona .tag { font-size: 0.85rem; color: #047857; font-weight: 600; margin-bottom: 0.5rem; }
  .card-persona p { margin: 0; font-size: 0.88rem; color: #4b5563; line-height: 1.45; }
  .loyalty-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #f59e0b44;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin: 1.25rem 0;
  }
  .loyalty-box h3 { margin: 0 0 0.5rem 0; color: #92400e; font-size: 1.1rem; }
  .loyalty-box p { margin: 0; color: #78350f; font-size: 0.95rem; line-height: 1.5; }
  /* Hide Streamlit header toolbar / command palette (typing there can open “Clear caches”, etc.) */
  [data-testid="stToolbar"] { display: none !important; }
</style>
"""


def _md5_path(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _power_bi_embed_url() -> str:
    try:
        if "POWERBI_REPORT_EMBED_URL" in st.secrets:
            u = str(st.secrets["POWERBI_REPORT_EMBED_URL"]).strip()
            if u.startswith(("http://", "https://")):
                return u
    except Exception:
        pass
    return ""


def _sample_data_candidates() -> list[Path]:
    return [
        _ROOT / "artifacts" / "marketing_campaign.csv",
        _ROOT / "notebooks" / "data" / "marketing_campaign.csv",
    ]


def _existing_sample_data_path() -> Path | None:
    for path in _sample_data_candidates():
        if path.is_file():
            return path
    return None


def _quality_story(silhouette: float) -> tuple[str, str]:
    """Plain-language group clarity (internal number is not shown by default)."""
    if silhouette >= 0.4:
        return "Very clear groups", "Shoppers in each group look alike; groups look different from each other — strong basis for targeted offers."
    if silhouette >= 0.28:
        return "Clear enough to use", "Good separation for real campaigns and loyalty tiers. Most retailers would be comfortable acting on this."
    if silhouette >= 0.15:
        return "Usable with care", "Some overlap between groups — combine with your business judgment when choosing messages."
    return "Overlapping groups", "Consider fewer groups or richer data before major budget decisions."


def _fmt_value(value: float, money: bool = False) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if money:
        return f"${value:,.0f}" if abs(value) >= 100 else f"${value:,.2f}"
    return f"{value:,.1f}" if abs(value) < 1000 else f"{value:,.0f}"


def _build_run_story(
    seg: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    top_id: int,
    top_name: str,
) -> tuple[str, str]:
    total_customers = len(seg)
    top_count = int((seg["cluster_id"] == top_id).sum())
    top_pct = 100 * top_count / total_customers if total_customers else 0

    parts = [
        f"The highest-value segment in this run is {top_name}, covering about {top_pct:.1f}% of the uploaded rows."
    ]

    if "monetary" in seg.columns:
        total_spend = float(seg["monetary"].sum())
        top_spend = float(seg.loc[seg["cluster_id"] == top_id, "monetary"].sum())
        overall_avg_spend = float(seg["monetary"].mean()) if len(seg) else 0.0
        top_avg_spend = float(seg.loc[seg["cluster_id"] == top_id, "monetary"].mean())
        spend_share = 100 * top_spend / total_spend if total_spend else 0.0
        multiplier = top_avg_spend / overall_avg_spend if overall_avg_spend else 0.0
        parts.append(
            f"It contributes about {spend_share:.1f}% of total spend in the file, with average spend {_fmt_value(top_avg_spend, money=True)} per customer"
            f" ({multiplier:.2f}x the file average)."
        )

    top_row = cluster_summary.loc[cluster_summary["cluster_id"] == top_id].iloc[0]
    if "frequency" in cluster_summary.columns:
        parts.append(f"Average purchase frequency for this segment is {_fmt_value(top_row['frequency'])}.")
    if "total_accepted_cmp" in cluster_summary.columns:
        parts.append(f"Average campaign acceptance is {_fmt_value(top_row['total_accepted_cmp'])}, which helps identify a loyalty target.")

    return "Dynamic loyalty opportunity", " ".join(parts)


def _schema_note(load_mode: str) -> tuple[str, str]:
    if load_mode == "marketing":
        return (
            "Retail schema detected",
            "This upload matches the original marketing-style customer table, so the app can compute retail features like spend, frequency, campaign acceptance, and shopper personas.",
        )
    return (
        "Generic numeric clustering",
        "This upload did not match the full retail schema. The app still clustered the numeric columns it found, but retail-specific engineered features and named personas are simplified.",
    )


@st.cache_data(show_spinner=True)
def analyze_store_customers(content_signature: str, data_path: str, n_groups: int, seed: int):
    """content_signature forces cache refresh when file bytes change."""
    _ = content_signature  # part of cache key only
    return run_segmentation(
        data_path=Path(data_path),
        n_clusters=int(n_groups),
        n_components=3,
        random_state=int(seed),
    )


def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    default_path = _existing_sample_data_path()

    st.markdown(
        """
<div class="top-banner">
  <h1>🛍️ Retail Store Customer Segmentation &amp; Profiling</h1>
  <p class="sub">Turn your customer spreadsheet into <strong>clear shopper groups</strong> — who spends most, who shops online, who needs a nudge — 
  so you can target marketing and design a loyalty program (like <strong>Prosperous</strong> for your best customers).</p>
  <p class="fine">Research title: <em>Optimizing intra-cluster compactness &amp; inter-cluster separation — K-Means for retail analytics</em></p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
**What this does (simple):**  
It does **not** guess “will they buy tomorrow.” It **sorts customers into a few similar groups** from their past behaviour and profile — 
like “premium in-store buyers” vs “deal hunters on the web.” Same idea as **recency, frequency, and spend** in retail, without you doing the math.
"""
    )

    if "data_path" not in st.session_state:
        st.session_state.data_path = str(default_path.resolve()) if default_path is not None else ""
        st.session_state.n_groups = 5
        st.session_state.content_sig = _md5_path(Path(st.session_state.data_path)) if default_path is not None else ""

    try:
        input_wrap = st.container(border=True)
    except TypeError:
        input_wrap = st.container()
    with input_wrap:
        st.markdown("### Your input")
        c1, c2, c3 = st.columns((1.1, 0.9, 0.7))
        with c1:
            up = st.file_uploader(
                "Customer data file",
                type=["csv", "txt"],
                help="Comma, tab, semicolon, or pipe separated. Same information as the sample file (ID, spend, channels, date joined, …).",
            )
        with c2:
            _opts = [3, 4, 5, 6]
            _def = st.session_state.n_groups if st.session_state.n_groups in _opts else 5
            n_groups = st.selectbox(
                "How many shopper types to find?",
                options=_opts,
                index=_opts.index(_def),
                help="Five types match the original project (Prosperous, Web-Shrewd, …). Click **Run analysis** after you change this.",
            )
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("Run analysis", type="primary", use_container_width=True)

        st.caption(
            "Using the **sample store dataset** until you upload your own file. "
            "Click **Run analysis** after you change the file or the number of groups."
        )

    if run:
        if up is not None:
            dest = UPLOAD_DIR / "uploaded_customers.csv"
            dest.write_bytes(up.getvalue())
            st.session_state.data_path = str(dest.resolve())
        else:
            if default_path is None:
                st.error("Could not find the bundled sample data file in `artifacts/` or `notebooks/data/`.")
                st.stop()
            st.session_state.data_path = str(default_path.resolve())
        st.session_state.n_groups = int(n_groups)
        st.session_state.content_sig = _md5_path(Path(st.session_state.data_path))
        st.rerun()

    data_path = Path(st.session_state.data_path)
    if not data_path.is_file():
        st.error(
            "Could not find customer data. Add `marketing_campaign.csv` under `artifacts/` or `notebooks/data/`, or upload a file."
        )
        st.stop()

    with st.spinner("Grouping customers by behaviour and profile…"):
        try:
            result = analyze_store_customers(
                st.session_state.content_sig,
                str(data_path.resolve()),
                st.session_state.n_groups,
                42,
            )
        except (ValueError, CustomException, Exception) as first_error:
            fallback_candidates = [
                p for p in _sample_data_candidates()
                if p.is_file() and p.resolve() != data_path.resolve()
            ]
            if str(data_path.resolve()) not in {str(p.resolve()) for p in _sample_data_candidates()} or not fallback_candidates:
                if isinstance(first_error, ValueError):
                    st.error(str(first_error))
                elif isinstance(first_error, CustomException):
                    st.error(str(first_error))
                else:
                    st.error(f"Could not read or analyze this file. Check the format and columns. ({first_error})")
                st.stop()

            fallback_path = fallback_candidates[0]
            try:
                result = analyze_store_customers(
                    _md5_path(fallback_path),
                    str(fallback_path.resolve()),
                    st.session_state.n_groups,
                    42,
                )
                st.warning(
                    f"Primary sample file failed to load; using fallback sample dataset at `{fallback_path}` instead."
                )
                st.session_state.data_path = str(fallback_path.resolve())
                st.session_state.content_sig = _md5_path(fallback_path)
            except Exception:
                if isinstance(first_error, ValueError):
                    st.error(str(first_error))
                elif isinstance(first_error, CustomException):
                    st.error(str(first_error))
                else:
                    st.error(f"Could not read or analyze this file. Check the format and columns. ({first_error})")
                st.stop()

    try:
        write_powerbi_csvs(result, _ROOT / "powerbi" / "exports")
    except OSError:
        pass

    persona_map = persona_labels_for_clusters(result.cluster_summary, st.session_state.n_groups)
    seg = result.customer_segment.copy()
    seg["Shopper type"] = apply_persona_column(seg, persona_map)

    sil = float(result.metrics["silhouette_score"])
    quality_title, quality_text = _quality_story(sil)
    total_customers = len(seg)
    rank_col = business_rank_column(result.cluster_summary)
    ranked = result.cluster_summary.sort_values(rank_col, ascending=False)
    cs = result.cluster_summary.copy()
    cs["Shopper type"] = cs["cluster_id"].map(lambda x: persona_map.get(int(x), f"Group {x}"))
    top_id = int(ranked.iloc[0]["cluster_id"])
    top_name = persona_map[top_id]
    top_count = int((seg["cluster_id"] == top_id).sum())
    top_pct_people = 100 * top_count / total_customers
    if "monetary" in seg.columns:
        total_metric = float(seg["monetary"].sum())
        top_metric = float(seg.loc[seg["cluster_id"] == top_id, "monetary"].sum())
        top_share = top_metric / total_metric * 100 if total_metric else 0
        share_label = "Share of spend — top group"
    else:
        top_share = top_pct_people
        share_label = "Share of rows — top group"

    st.markdown("### Your results")
    schema_title, schema_text = _schema_note(result.load_mode)
    st.markdown(
        f'<div class="kpi-row">'
        f'<div class="kpi"><div class="label">Rows analyzed</div><div class="value">{total_customers:,}</div>'
        f'<div class="hint">Each row is one record from your file (customer or line item, depending on data).</div></div>'
        f'<div class="kpi"><div class="label">Groups found</div><div class="value">{st.session_state.n_groups}</div>'
        f'<div class="hint">Distinct segments from your numeric columns.</div></div>'
        f'<div class="kpi"><div class="label">How distinct are the groups?</div><div class="value">{quality_title}</div>'
        f'<div class="hint">{quality_text}</div></div>'
        f'<div class="kpi"><div class="label">{share_label}</div><div class="value">{top_share:.0f}%</div>'
        f'<div class="hint"><strong>{top_name}</strong> — about {top_pct_people:.1f}% of rows (ranked using: {rank_col}).</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.caption(f"**{schema_title}:** {schema_text}")

    loyalty_title, loyalty_text = _build_run_story(seg, result.cluster_summary, top_id, top_name)
    st.markdown(
        f"""
<div class="loyalty-box">
  <h3>⭐ {loyalty_title}</h3>
  <p>{loyalty_text}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("Who are these groups?")
    st.caption("Plain-English profiles — use them in meetings, not equations.")

    order_ids = [int(x) for x in ranked["cluster_id"].tolist()]
    row_start = 0
    while row_start < len(order_ids):
        batch = order_ids[row_start : row_start + 3]
        cols = st.columns(len(batch))
        for col, cid in zip(cols, batch):
            name = persona_map[cid]
            cluster_row = cs.loc[cs["cluster_id"] == cid].iloc[0]
            tag, body = dynamic_story_for_cluster(
                cluster_row=cluster_row,
                cluster_summary=cs,
                cluster_name=name,
                load_mode=result.load_mode,
            )
            with col:
                st.markdown(
                    f'<div class="card-persona"><div class="tag">{tag}</div><h4>{name}</h4><p>{body}</p></div>',
                    unsafe_allow_html=True,
                )
        row_start += 3

    st.subheader("Charts")
    f1 = px.bar(
        cs.sort_values("customer_count", ascending=True),
        x="customer_count",
        y="Shopper type",
        orientation="h",
        title="How many customers are in each group?",
        color_discrete_sequence=["#2d6a4f"],
        height=max(320, 70 * len(cs)),
    )
    f1.update_layout(showlegend=False, yaxis_title=None)
    st.plotly_chart(f1, use_container_width=True)

    bar_metric = rank_col if rank_col in cs.columns else business_rank_column(cs)
    if bar_metric not in cs.columns:
        num_cols = [c for c in cs.columns if c not in ("Shopper type", "cluster_id", "customer_count")]
        bar_metric = num_cols[0] if num_cols else "cluster_id"
    f2 = px.bar(
        cs.sort_values(bar_metric, ascending=True),
        x=bar_metric,
        y="Shopper type",
        orientation="h",
        title=f"Average {bar_metric} per group",
        color_discrete_sequence=["#40916c"],
        height=max(320, 70 * len(cs)),
    )
    f2.update_layout(showlegend=False, yaxis_title=None)
    st.plotly_chart(f2, use_container_width=True)

    if {"pca_1", "pca_2", "pca_3"}.issubset(seg.columns):
        f3 = px.scatter_3d(
            seg,
            x="pca_1",
            y="pca_2",
            z="pca_3",
            color="Shopper type",
            title="Customer map (each dot is a shopper — similar people sit closer together)",
            labels={"pca_1": "Shopping pattern A", "pca_2": "Shopping pattern B", "pca_3": "Shopping pattern C"},
            opacity=0.65,
            height=640,
        )
        st.plotly_chart(f3, use_container_width=True)
        st.caption("Axes are combined behaviour scores — not single columns like income. They help **visualize** the groups.")
    elif {"pca_1", "pca_2"}.issubset(seg.columns):
        f3 = px.scatter(
            seg,
            x="pca_1",
            y="pca_2",
            color="Shopper type",
            title="Customer map (2D fallback)",
            labels={"pca_1": "Shopping pattern A", "pca_2": "Shopping pattern B"},
            opacity=0.7,
            height=560,
        )
        st.plotly_chart(f3, use_container_width=True)
        st.caption("Only two PCA dimensions were available for this upload, so the customer map is shown in 2D.")

    st.subheader("Download segmented customers")
    dl = seg.drop(columns=[c for c in ("pca_1", "pca_2", "pca_3") if c in seg.columns], errors="ignore")
    dl = dl.rename(columns={"cluster_id": "Group number"})
    csv_bytes = dl.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download spreadsheet (each customer + shopper type)",
        data=csv_bytes,
        file_name="customers_with_shopper_types.csv",
        mime="text/csv",
    )

    with st.expander("Technical details (for teachers / report — optional)"):
        st.markdown(
            f"- **Silhouette (internal):** `{sil:.4f}` — statistical check that groups are separated.\n"
            f"- **Method:** grouping algorithm with **{st.session_state.n_groups}** groups, three summary dimensions, seed 42.\n"
            f"- **Power BI / exports:** CSV files are also saved under `powerbi/exports/` for Microsoft dashboards if needed."
        )
        st.dataframe(result.metrics_table, use_container_width=True, hide_index=True)
        sk = result.silhouette_by_k
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=sk["k"], y=sk["silhouette_score"], name="Separation score", yaxis="y1"))
        fig_t.add_trace(go.Scatter(x=sk["k"], y=sk["inertia"], name="Tightness (internal)", yaxis="y2"))
        fig_t.update_layout(
            title="Trying different numbers of groups (internal diagnostic)",
            xaxis_title="Number of groups",
            yaxis_title="Separation",
            yaxis2=dict(title="Tightness", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig_t, use_container_width=True)


if __name__ == "__main__":
    main()
