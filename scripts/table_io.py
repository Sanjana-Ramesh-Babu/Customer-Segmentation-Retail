"""
Load tabular customer files with automatic delimiter / encoding handling and
column-name normalization (regex + aliases) for the marketing-campaign schema.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import BinaryIO, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Canonical names expected by scripts.modelling_utils.feature_engineering (after its own .lower())
_REQUIRED_FOR_PIPELINE = frozenset(
    {
        "id",
        "year_birth",
        "education",
        "marital_status",
        "income",
        "kidhome",
        "teenhome",
        "dt_customer",
        "recency",
        "mntwines",
        "mntfruits",
        "mntmeatproducts",
        "mntfishproducts",
        "mntsweetproducts",
        "mntgoldprods",
        "numdealspurchases",
        "numwebpurchases",
        "numcatalogpurchases",
        "numstorepurchases",
        "numwebvisitsmonth",
        "acceptedcmp1",
        "acceptedcmp2",
        "acceptedcmp3",
        "acceptedcmp4",
        "acceptedcmp5",
        "complain",
        "response",
        "z_costcontact",
        "z_revenue",
    }
)

_COLUMN_ALIASES: dict[str, str] = {
    "customer_id": "id",
    "cust_id": "id",
    "client_id": "id",
    "userid": "id",
    "user_id": "id",
    "birth_year": "year_birth",
    "birthyear": "year_birth",
    "yob": "year_birth",
    "customer_since": "dt_customer",
    "join_date": "dt_customer",
    "joindate": "dt_customer",
    "first_purchase_date": "dt_customer",
    "maritalstatus": "marital_status",
    "marital": "marital_status",
    "mnt_wines": "mntwines",
    "mnt_fruits": "mntfruits",
    "mnt_meat_products": "mntmeatproducts",
    "mntmeat_products": "mntmeatproducts",
    "mnt_fish_products": "mntfishproducts",
    "mnt_sweet_products": "mntsweetproducts",
    "mnt_gold_prods": "mntgoldprods",
    "mntgold_prods": "mntgoldprods",
    "num_deals_purchases": "numdealspurchases",
    "num_web_purchases": "numwebpurchases",
    "num_catalog_purchases": "numcatalogpurchases",
    "num_store_purchases": "numstorepurchases",
    "num_web_visits_month": "numwebvisitsmonth",
    "web_visits_per_month": "numwebvisitsmonth",
    "kid_home": "kidhome",
    "teen_home": "teenhome",
    "accepted_cmp1": "acceptedcmp1",
    "accepted_cmp_1": "acceptedcmp1",
    "accepted_cmp2": "acceptedcmp2",
    "accepted_cmp_2": "acceptedcmp2",
    "accepted_cmp3": "acceptedcmp3",
    "accepted_cmp_3": "acceptedcmp3",
    "accepted_cmp4": "acceptedcmp4",
    "accepted_cmp_4": "acceptedcmp4",
    "accepted_cmp5": "acceptedcmp5",
    "accepted_cmp_5": "acceptedcmp5",
    "zcostcontact": "z_costcontact",
    "z_cost_contact": "z_costcontact",
    "zrevenue": "z_revenue",
}


def slugify_column(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^\w\s\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _apply_regex_canonical(slug: str) -> str:
    m = re.match(r"^accepted[_\s]?cmp[_\s]?([1-5])$", slug, re.I)
    if m:
        return f"acceptedcmp{m.group(1)}"
    fixed: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"^mnt[_\s]?wines?$", re.I), "mntwines"),
        (re.compile(r"^mnt[_\s]?fruits?$", re.I), "mntfruits"),
        (re.compile(r"^mnt[_\s]?meat[_\s]?products?$", re.I), "mntmeatproducts"),
        (re.compile(r"^mnt[_\s]?fish[_\s]?products?$", re.I), "mntfishproducts"),
        (re.compile(r"^mnt[_\s]?sweet[_\s]?products?$", re.I), "mntsweetproducts"),
        (re.compile(r"^mnt[_\s]?gold[_\s]?prods?$", re.I), "mntgoldprods"),
        (re.compile(r"^num[_\s]?deals[_\s]?purchases?$", re.I), "numdealspurchases"),
        (re.compile(r"^num[_\s]?web[_\s]?purchases?$", re.I), "numwebpurchases"),
        (re.compile(r"^num[_\s]?catalog[_\s]?purchases?$", re.I), "numcatalogpurchases"),
        (re.compile(r"^num[_\s]?store[_\s]?purchases?$", re.I), "numstorepurchases"),
        (re.compile(r"^num[_\s]?web[_\s]?visits[_\s]?month$", re.I), "numwebvisitsmonth"),
        (re.compile(r"^dt[_\s]?customer$", re.I), "dt_customer"),
        (re.compile(r"^year[_\s]?birth$", re.I), "year_birth"),
        (re.compile(r"^marital[_\s]?status$", re.I), "marital_status"),
        (re.compile(r"^kid[_\s]?home$", re.I), "kidhome"),
        (re.compile(r"^teen[_\s]?home$", re.I), "teenhome"),
        (re.compile(r"^(customer|client)[_\s]?id$", re.I), "id"),
    ]
    for pat, repl in fixed:
        if pat.match(slug):
            return repl
    return slug


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to names expected by feature_engineering."""
    out = df.copy()
    rename_map: dict[str, str] = {}
    inverse: dict[str, list[str]] = {}
    for col in out.columns:
        slug = slugify_column(col)
        slug = _apply_regex_canonical(slug)
        slug = _COLUMN_ALIASES.get(slug, slug)
        slug = _COLUMN_ALIASES.get(slug, slug)
        rename_map[col] = slug
        inverse.setdefault(slug, []).append(str(col))
    conflicts = {k: v for k, v in inverse.items() if len(v) > 1}
    if conflicts:
        raise ValueError(
            "Multiple columns map to the same field name after cleanup; keep one column per role: "
            + str(conflicts)
        )
    out.rename(columns=rename_map, inplace=True)
    return out


def validate_pipeline_columns(df: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_FOR_PIPELINE - set(df.columns))
    if missing:
        raise ValueError(
            "These required columns are missing (or could not be matched from your headers): "
            + ", ".join(missing)
            + ". This pipeline expects **one row per customer** like `marketing_campaign.csv` "
            "(ID, date joined, spend by category, channel counts, campaigns, etc.). "
            "Transaction-only exports (e.g. line-item retail logs) use different columns and are not supported here."
        )


def _encoding_probe_chunks(raw: bytes, max_total: int = 600_000) -> bytes:
    """Start + middle + end so a bad byte (e.g. 0xa3 as £) is not missed when it’s past the first block."""
    n = len(raw)
    if n <= max_total:
        return raw
    part = max_total // 3
    return raw[:part] + raw[n // 2 : n // 2 + part] + raw[-part:]


def _detect_encoding(raw: bytes) -> str:
    """
    Pick an encoding suitable for the whole file (handles £ as 0xa3 in Latin-1 / cp1252, etc.).
    """
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None and getattr(best, "encoding", None):
            return str(best.encoding)
    except Exception:
        pass
    probe = _encoding_probe_chunks(raw)
    for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1", "latin-1"):
        try:
            probe.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _sniff_delimiter(sample: str) -> str:
    sample = sample.lstrip("\ufeff")
    if not sample.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        if dialect.delimiter in ",\t;|":
            return dialect.delimiter
    except csv.Error:
        pass
    lines = [ln for ln in sample.splitlines()[:8] if ln.strip()]
    if not lines:
        return ","
    counts = {d: sum(line.count(d) for line in lines) for d in (",", "\t", ";", "|")}
    return max(counts, key=counts.get)


def read_tabular_bytes(raw: bytes) -> pd.DataFrame:
    """
    Read CSV/TSV-like bytes: detect encoding (UTF-8, Windows-1252, Latin-1, …),
    sniff delimiter, then let pandas decode the full file (avoids UTF-8 errors on £ etc.).
    """
    enc = _detect_encoding(raw)
    sample = raw[: min(len(raw), 500_000)]
    try:
        sample_text = sample.decode(enc, errors="strict")
    except UnicodeDecodeError:
        enc = "latin-1"
        sample_text = sample.decode(enc, errors="strict")
    sep = _sniff_delimiter(sample_text[:20000])
    last_err: Exception | None = None
    for attempt in (enc, "cp1252", "latin-1", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                engine="python",
                encoding=attempt,
                encoding_errors="replace",
            )
        except (UnicodeDecodeError, LookupError, ValueError) as e:
            last_err = e
            continue
    raise OSError(f"Could not decode CSV with supported encodings. Last error: {last_err}") from last_err


def read_tabular_path(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    return read_tabular_bytes(path.read_bytes())


def _unique_slug_rename(columns: list[str]) -> dict[str, str]:
    """Map original headers to unique snake_case names (no marketing alias table)."""
    seen: dict[str, int] = {}
    out: dict[str, str] = {}
    for col in columns:
        base = slugify_column(col) or "column"
        name = base
        if name in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out[col] = name
    return out


_ID_SLUGS = frozenset(
    {
        "id",
        "customer_id",
        "client_id",
        "customerid",
        "userid",
        "user_id",
        "cust_id",
        "invoiceno",
        "invoice_no",
        "transaction_id",
    }
)


def prepare_generic_numeric_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a numeric-only feature matrix from any tabular file: slug headers, coerce numbers,
    pick an ID column if present, else row index.
    """
    out = df.copy()
    out.rename(columns=_unique_slug_rename(list(out.columns)), inplace=True)

    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    id_col = None
    for c in out.columns:
        if c in _ID_SLUGS or (c.endswith("_id") and c != "cluster_id"):
            id_col = c
            break

    if id_col is not None:
        ids = out[id_col].astype(str)
        ids.name = "customer_id"
        features = out.drop(columns=[id_col])
    else:
        ids = pd.Series(np.arange(len(out), dtype=np.int64), name="customer_id")
        features = out

    num = features.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.dropna(axis=1, how="all")
    keep = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
    num = num[keep]

    if num.shape[1] < 1:
        raise ValueError(
            "No usable numeric columns after cleaning. Add numeric fields or export with numbers parsed (not text)."
        )

    num = num.astype(np.float64)
    return num, ids


def load_for_segmentation(
    path: Optional[Union[str, Path]] = None,
    raw: Optional[Union[bytes, BinaryIO]] = None,
) -> tuple[pd.DataFrame, Optional[pd.Series], Literal["marketing", "generic"]]:
    """
    Load CSV/TSV: use full marketing schema + feature_engineering when possible;
    otherwise cluster on all numeric columns dynamically.
    """
    if raw is not None:
        data = raw.read() if hasattr(raw, "read") else raw
        base = read_tabular_bytes(data)
    elif path is not None:
        base = read_tabular_path(path)
    else:
        raise ValueError("Provide path= or raw=.")

    marketing_df: Optional[pd.DataFrame] = None
    try:
        marketing_df = canonicalize_columns(base.copy())
    except ValueError:
        marketing_df = None

    if marketing_df is not None and _REQUIRED_FOR_PIPELINE <= set(marketing_df.columns):
        return marketing_df, None, "marketing"

    cdf, ids = prepare_generic_numeric_frame(base)
    return cdf, ids, "generic"


def load_customer_marketing_table(
    path: Optional[Union[str, Path]] = None,
    raw: Optional[Union[bytes, BinaryIO]] = None,
) -> pd.DataFrame:
    """Strict: same as before — marketing schema only (raises if generic)."""
    df, _, mode = load_for_segmentation(path=path, raw=raw)
    if mode != "marketing":
        raise ValueError(
            "File does not include all columns required for the marketing-campaign pipeline."
        )
    return df
