"""
CLI: export Power BI–ready CSVs to powerbi/exports/

Usage (from project root):
  python -m scripts.export_powerbi_datasets
  python -m scripts.export_powerbi_datasets --data notebooks/data/marketing_campaign.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.segmentation_pipeline import export_powerbi_csvs, run_segmentation


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        type=Path,
        default=root / "notebooks" / "data" / "marketing_campaign.csv",
        help="Customer table (.csv / .txt): comma, tab, semicolon, or pipe — same columns as sample",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "powerbi" / "exports",
        help="Output folder for CSV files",
    )
    p.add_argument("--k", type=int, default=5, help="Number of clusters")
    args = p.parse_args()

    result = run_segmentation(data_path=args.data, n_clusters=args.k)
    paths = export_powerbi_csvs(result, args.out)
    print(f"Exported {len(paths)} files to {args.out.resolve()}:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
