#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate CNR descriptors against UniProt Region annotations with MobiDB-lite-style disorder."""
import argparse, json, re
from pathlib import Path
import pandas as pd

def parse_disordered_regions(region_text):
    if pd.isna(region_text):
        return []
    regions = []
    for part in re.split(r'(?=REGION\s+)', str(region_text)):
        if not part.strip().startswith("REGION"):
            continue
        if re.search(r'/note="Disordered"', part, flags=re.I):
            m = re.search(r'REGION\s+(\d+)\.\.(\d+)', part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if b >= a:
                    regions.append((a, b))
    return regions

def union_length(intervals):
    if not intervals:
        return 0, 0, 0
    intervals = sorted(intervals)
    merged = []
    for a, b in intervals:
        if not merged or a > merged[-1][1] + 1:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    total = sum(b - a + 1 for a, b in merged)
    longest = max(b - a + 1 for a, b in merged)
    return total, len(merged), longest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnr-table", required=True)
    ap.add_argument("--uniprot-tsv", required=True)
    ap.add_argument("--outdir", default="cnr_uniprot_mobidb_validation")
    ap.add_argument("--n-cpu", type=int, default=1, help="Reserved; script runs serially.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cnr = pd.read_csv(args.cnr_table)
    uni = pd.read_csv(args.uniprot_tsv, sep="\t")
    if "Region" not in uni.columns:
        raise ValueError("UniProt TSV must include a 'Region' column.")

    uni["disordered_regions"] = uni["Region"].apply(parse_disordered_regions)
    tmp = uni["disordered_regions"].apply(union_length)
    uni["uniprot_disorder_length"] = tmp.apply(lambda x: x[0])
    uni["uniprot_disorder_region_count"] = tmp.apply(lambda x: x[1])
    uni["uniprot_longest_disorder_region"] = tmp.apply(lambda x: x[2])
    uni["Length"] = pd.to_numeric(uni["Length"], errors="coerce")
    uni["uniprot_disorder_fraction"] = uni["uniprot_disorder_length"] / uni["Length"]
    uni["has_uniprot_disorder"] = uni["uniprot_disorder_length"] > 0

    uni_keep = uni.rename(columns={"Entry": "UniProt_ACC", "Length": "uniprot_length"})[
        ["UniProt_ACC", "Entry Name", "Protein names", "Gene Names", "uniprot_length",
         "uniprot_disorder_length", "uniprot_disorder_region_count",
         "uniprot_longest_disorder_region", "uniprot_disorder_fraction", "has_uniprot_disorder"]
    ]

    merged = cnr.merge(uni_keep, on="UniProt_ACC", how="inner")
    merged["uniprot_disorder_class"] = pd.cut(
        merged["uniprot_disorder_fraction"].fillna(0),
        bins=[-0.001, 0, 0.1, 0.3, 0.6, 1.0],
        labels=["none", "low (0–0.1)", "moderate (0.1–0.3)", "high (0.3–0.6)", "very high (>0.6)"]
    )

    descriptors = ["rho_HNC","bethe_lambda_f","graph_lambda_f","R_topo","Delta_lambda_f",
                   "graph_max_m","graph_susceptibility_peak","mean_plddt","retained_fraction"]
    corrs = {}
    for col in descriptors:
        if col in merged.columns:
            sub = merged[[col, "uniprot_disorder_fraction"]].dropna()
            if len(sub) > 2:
                corrs[col] = float(sub.corr(method="spearman").iloc[0, 1])

    merged.to_csv(outdir / "AF_CNR_UniProt_MobiDBlite_merged_table.csv", index=False)
    pd.DataFrame({"descriptor": list(corrs.keys()),
                  "spearman_vs_uniprot_disorder_fraction": list(corrs.values())}).to_csv(
        outdir / "AF_CNR_UniProt_MobiDBlite_spearman_correlations.csv", index=False)

    class_summary = merged.groupby("uniprot_disorder_class", observed=True).agg(
        n=("protein_id", "count"),
        disorder_fraction_median=("uniprot_disorder_fraction", "median"),
        rho_HNC_median=("rho_HNC", "median"),
        bethe_lambda_f_median=("bethe_lambda_f", "median"),
        graph_lambda_f_median=("graph_lambda_f", "median"),
        R_topo_median=("R_topo", "median"),
        graph_max_m_median=("graph_max_m", "median"),
        chi_median=("graph_susceptibility_peak", "median"),
        retained_fraction_median=("retained_fraction", "median"),
        mean_plddt_median=("mean_plddt", "median")
    ).reset_index()
    class_summary.to_csv(outdir / "AF_CNR_UniProt_MobiDBlite_disorder_class_summary.csv", index=False)

    summary = {
        "uniprot_rows": int(len(uni)),
        "AF_CNR_rows": int(len(cnr)),
        "merged_rows": int(len(merged)),
        "merged_with_disorder_region": int(merged["has_uniprot_disorder"].sum()),
        "spearman_vs_uniprot_disorder_fraction": corrs,
    }
    (outdir / "AF_CNR_UniProt_MobiDBlite_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
